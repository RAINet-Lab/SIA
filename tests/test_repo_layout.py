from pathlib import Path


def test_repo_layout_exists() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / 'src/sia/core').exists()
    assert (repo_root / 'use_cases/abr').exists()
    assert (repo_root / 'scripts/smoke_imports.py').exists()
