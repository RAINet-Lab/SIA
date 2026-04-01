from __future__ import annotations

import argparse
import os
import shutil
import ssl
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from urllib.request import urlopen

NORWAY_TRACE_INDEX = 'https://datasets.simula.no/downloads/hsdpa-tcp-logs/'
NORWAY_ROUTE_PREFIXES = ('bus.', 'car.', 'ferry.', 'metro.', 'train.', 'tram.')
TEST_CHUNK_SECONDS = 320.0
TEST_CHUNK_STRIDE_SECONDS = 60.0

DATASETS = {
    'abr': {
        'status': 'public-download-available',
        'target': 'data/raw/abr',
        'notes': 'Downloads public Norway mobility traces, converts them into the legacy two-column format, and prepares compatibility links for the ABR code.',
    },
    'mimo': {
        'status': 'contact-authors',
        'target': 'data/raw/mimo',
        'notes': 'The public repo keeps the code/notebooks, but the original traces are not redistributed. See use_cases/mimo/README.md for the paper and contact guidance.',
    },
    'ran_slicing': {
        'status': 'confidential-dataset',
        'target': 'data/raw/ran_slicing',
        'notes': 'Only scripts/notebooks/results are shipped publicly. See use_cases/ran_slicing/README.md for contact guidance.',
    },
}


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != 'a':
            return
        for key, value in attrs:
            if key == 'href' and value:
                self.links.append(value)
                break


def fetch_text(url: str) -> str:
    context = ssl._create_unverified_context()
    with urlopen(url, context=context, timeout=60) as response:
        return response.read().decode('utf-8', errors='ignore')


def list_links(url: str) -> list[str]:
    parser = LinkExtractor()
    parser.feed(fetch_text(url))
    return parser.links


def download_file(url: str, destination: Path) -> None:
    context = ssl._create_unverified_context()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, context=context, timeout=60) as response, destination.open('wb') as handle:
        shutil.copyfileobj(response, handle)


def ensure_clean_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_link(link_path: Path, target_path: Path, *, force: bool) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if not force:
            return
        ensure_clean_path(link_path)
    relative_target = os.path.relpath(target_path, link_path.parent)
    os.symlink(relative_target, link_path)


def route_directories() -> list[str]:
    links = list_links(NORWAY_TRACE_INDEX)
    routes = []
    for link in links:
        cleaned = link.rstrip('/')
        if cleaned.startswith(NORWAY_ROUTE_PREFIXES):
            routes.append(cleaned)
    return sorted(set(routes))


def route_files(route: str) -> list[str]:
    route_url = urljoin(NORWAY_TRACE_INDEX, f'{route}/')
    files = []
    for link in list_links(route_url):
        if link.startswith('report.') and link.endswith('.log'):
            files.append(link)
    return sorted(set(files))


def convert_raw_trace(raw_path: Path, output_path: Path) -> None:
    first_timestamp: float | None = None
    converted: list[str] = []
    with raw_path.open() as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) < 6:
                continue
            timestamp = float(parts[0])
            payload_bytes = float(parts[4])
            recv_time_ms = float(parts[5])
            if recv_time_ms <= 0:
                continue
            if first_timestamp is None:
                first_timestamp = timestamp
            elapsed_seconds = timestamp - first_timestamp
            throughput_mbps = payload_bytes / recv_time_ms * 8.0 / 1000.0
            converted.append(f'{elapsed_seconds:.6f}\t{throughput_mbps:.12f}\n')
    if not converted:
        raise RuntimeError(f'failed to convert trace: {raw_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(converted))


def iter_trace_windows(trace_path: Path, *, chunk_seconds: float, stride_seconds: float) -> Iterable[list[tuple[float, float]]]:
    points: list[tuple[float, float]] = []
    with trace_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            time_str, bw_str = line.split()[:2]
            points.append((float(time_str), float(bw_str)))
    if not points:
        return []
    max_time = points[-1][0]
    windows: list[list[tuple[float, float]]] = []
    start = 0.0
    while start + chunk_seconds <= max_time + 1e-9:
        end = start + chunk_seconds
        chunk = [(time - start, bw) for time, bw in points if start <= time <= end]
        if len(chunk) >= 2:
            windows.append(chunk)
        start += stride_seconds
    return windows


def write_chunk(path: Path, chunk: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for time_value, bandwidth in chunk:
            handle.write(f'{time_value:.6f}\t{bandwidth:.12f}\n')


def bootstrap_abr(repo_root: Path, *, force: bool, create_only: bool, max_files: int | None) -> None:
    abr_root = repo_root / 'data' / 'raw' / 'abr'
    raw_root = abr_root / 'norway_raw'
    train_root = abr_root / 'train_all_files'
    test_root = abr_root / 'test_all_files'
    for path in (raw_root, train_root, test_root):
        path.mkdir(parents=True, exist_ok=True)

    if create_only:
        print('[abr] create-only mode; skipped downloads and conversion')
        prepare_abr_links(repo_root, train_root, test_root, force=force)
        return

    downloaded = 0
    converted_files: list[Path] = []
    route_dirs = route_directories()
    for route in route_dirs:
        for filename in route_files(route):
            if max_files is not None and downloaded >= max_files:
                break
            route_raw_dir = raw_root / route
            route_raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = route_raw_dir / filename
            if force or not raw_path.exists():
                source_url = urljoin(NORWAY_TRACE_INDEX, f'{route}/{filename}')
                print(f'[abr] download {source_url}')
                download_file(source_url, raw_path)
            converted_name = f'{route}-{filename}'
            converted_path = train_root / converted_name
            if force or not converted_path.exists():
                convert_raw_trace(raw_path, converted_path)
            converted_files.append(converted_path)
            downloaded += 1
        if max_files is not None and downloaded >= max_files:
            break

    if force:
        for stale in sorted(test_root.glob('*')):
            if stale.is_file() or stale.is_symlink():
                stale.unlink()

    chunk_counter = 1
    bus_sources = [path for path in converted_files if path.name.startswith('bus.')]
    chunk_sources = bus_sources or converted_files
    for source in chunk_sources:
        for chunk in iter_trace_windows(source, chunk_seconds=TEST_CHUNK_SECONDS, stride_seconds=TEST_CHUNK_STRIDE_SECONDS):
            chunk_path = test_root / f'norway_eval_{chunk_counter:03d}.log'
            write_chunk(chunk_path, chunk)
            chunk_counter += 1

    print(f'[abr] prepared {len(converted_files)} converted training traces in {train_root}')
    print(f'[abr] prepared {chunk_counter - 1} evaluation chunks in {test_root}')
    if max_files is not None:
        print('[abr] max-files limit was enabled for smoke testing; rerun without --max-files for the full public bootstrap')

    prepare_abr_links(repo_root, train_root, test_root, force=force)


def prepare_abr_links(repo_root: Path, train_root: Path, test_root: Path, *, force: bool) -> None:
    compatibility_links = {
        repo_root / 'pensive' / 'train_all_files': train_root,
        repo_root / 'pensive' / 'test_all_files': test_root,
        repo_root / 'use_cases' / 'abr' / 'vanilla' / 'train': train_root,
        repo_root / 'use_cases' / 'abr' / 'vanilla' / 'test': test_root,
        repo_root / 'use_cases' / 'abr' / 'lumos' / 'train': train_root,
        repo_root / 'use_cases' / 'abr' / 'lumos' / 'test': test_root,
        repo_root / 'use_cases' / 'abr' / 'xatu' / 'train': train_root,
        repo_root / 'use_cases' / 'abr' / 'xatu' / 'test': test_root,
    }
    for link_path, target_path in compatibility_links.items():
        ensure_link(link_path, target_path, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare public dataset directories for SIA.')
    parser.add_argument('--dataset', choices=['all', *DATASETS.keys()], default='all', help='Which dataset policy/bootstrap to run.')
    parser.add_argument('--create-only', action='store_true', help='Create expected dataset directories without downloading data.')
    parser.add_argument('--force', action='store_true', help='Replace existing generated compatibility links and bootstrap outputs.')
    parser.add_argument('--max-files', type=int, default=None, help='Limit ABR downloads for smoke testing.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    selected = DATASETS.keys() if args.dataset == 'all' else [args.dataset]
    for name in selected:
        meta = DATASETS[name]
        target = repo_root / meta['target']
        target.mkdir(parents=True, exist_ok=True)
        print(f'[{name}] target={target}')
        print(f'  status={meta["status"]}')
        print(f'  notes={meta["notes"]}')
        if name == 'abr':
            bootstrap_abr(repo_root, force=args.force, create_only=args.create_only, max_files=args.max_files)
        elif args.create_only:
            print('  create-only mode; nothing else to do')
        else:
            print('  download step intentionally not implemented for this dataset policy')


if __name__ == '__main__':
    main()
