"""
SIA Dashboard Exporter — CLI.

Reads a symbolic CSV (the output shape of the SIA symbolizer notebooks) and an
optional raw-KPI CSV, runs the dashboard compute helpers, and writes the fixture
bundle that the FastAPI webapp consumes.

Usage
-----
    python -m sia.dashboard.export \\
        --symbolic   data/raw/dashboard/ran_slicing/sample/symbolic.csv \\
        --kpi-raw    data/raw/dashboard/ran_slicing/sample/kpi_raw.csv \\
        --use-case   ran_slicing \\
        --agent      A3-R \\
        --out-dir    data/processed/dashboard/ran_slicing/sample

Output layout
-------------
    <out-dir>/
        meta.json
        symbolic.parquet
        kpi_raw.parquet          (if --kpi-raw supplied)
        forecast.parquet         (if --forecast supplied; has_forecast=True in meta)
        influence_score.parquet
        refinement_events.parquet
        mi.json

Notebooks are never imported or modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sia.dashboard.compute import (
    compute_influence_scores,
    compute_mutual_information,
    extract_refinement_events,
)

# ── Per-use-case configuration ────────────────────────────────────────────────

_USE_CASE_CONFIGS: dict[str, dict] = {
    "ran_slicing": {
        "column_map": {
            "tx_brate downlink [Mbps]": "tx_brate",
            "tx_pkts downlink":          "tx_pkts",
            "dl_buffer [bytes]":         "dl_buffer",
            "slice_prb":                 "prb_decision",
            "scheduling_policy":         "sched_decision",
        },
        "kpi_cols":    ["tx_brate", "tx_pkts", "dl_buffer"],
        "action_cols": ["prb_decision", "sched_decision"],
        "action_join": " | ",
        "slice_names": {0: "eMBB", 1: "mMTC", 2: "URLLC"},
    },
    "abr": {
        "column_map":  {},   # fixture writes short names directly
        "kpi_cols":    ["bandwidth", "throughput", "delay"],
        "action_cols": ["quality_level"],
        "action_join": None,  # single column → no join needed
        "slice_names": {0: "session_0", 1: "session_1", 2: "session_2"},
    },
    "mimo": {
        "column_map":  {},
        "kpi_cols":    ["mase", "jfi", "sinr"],
        "action_cols": ["beam_selection"],
        "action_join": None,
        "slice_names": {u: f"UE_{u}" for u in range(7)},
    },
}


def _normalise_symbolic(
    df: pd.DataFrame,
    column_map: dict[str, str],
    kpi_cols: list[str],
    action_cols: list[str],
    action_join: str | None,
) -> pd.DataFrame:
    df = df.rename(columns=column_map)

    if "timestep" not in df.columns:
        if "timestamp" in df.columns:
            df = df.sort_values(["timestamp", "slice_id"]).reset_index(drop=True)
            ts_index = {ts: i + 1 for i, ts in enumerate(df["timestamp"].unique())}
            df["timestep"] = df["timestamp"].map(ts_index)
        else:
            df["timestep"] = (df.index // df["slice_id"].nunique()) + 1

    if "action_combined" not in df.columns:
        if action_join and len(action_cols) > 1:
            df["action_combined"] = df[action_cols[0]].astype(str)
            for ac in action_cols[1:]:
                df["action_combined"] = df["action_combined"] + action_join + df[ac].astype(str)
        elif action_cols:
            df["action_combined"] = df[action_cols[0]].astype(str)

    for col, default in [
        ("reward", 0.0),
        ("able_to_improve", False),
        ("alternative", None),
        ("training", True),
    ]:
        if col not in df.columns:
            df[col] = default

    keep = ["timestep", "slice_id"] + kpi_cols + action_cols + [
        "action_combined", "reward", "able_to_improve", "alternative", "training"
    ]
    return df[[c for c in keep if c in df.columns]]


def export_run(
    symbolic_csv: str | Path,
    kpi_raw_csv: str | Path | None,
    use_case: str,
    agent: str,
    out_dir: str | Path,
    forecast_csv: str | Path | None = None,
    slice_names: dict[int, str] | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = _USE_CASE_CONFIGS.get(use_case, _USE_CASE_CONFIGS["ran_slicing"])
    kpi_cols    = cfg["kpi_cols"]
    action_cols = cfg["action_cols"]
    action_join = cfg["action_join"]
    snames      = slice_names or cfg["slice_names"]

    # ── Load & normalise symbolic data ──────────────────────────────────────
    sym_df = pd.read_csv(symbolic_csv)
    sym_df = _normalise_symbolic(sym_df, cfg["column_map"], kpi_cols, action_cols, action_join)
    sym_df.to_parquet(out / "symbolic.parquet", index=False)
    print(f"  symbolic.parquet  → {len(sym_df)} rows ({sym_df['slice_id'].nunique()} slices)")

    # ── Raw KPI time series (optional) ──────────────────────────────────────
    if kpi_raw_csv:
        raw_df = pd.read_csv(kpi_raw_csv)
        if "timestep" not in raw_df.columns and "timestamp" in raw_df.columns:
            raw_df = raw_df.sort_values(["timestamp", "slice_id"]).reset_index(drop=True)
            ts_index = {ts: i + 1 for i, ts in enumerate(raw_df["timestamp"].unique())}
            raw_df["timestep"] = raw_df["timestamp"].map(ts_index)
        raw_df.to_parquet(out / "kpi_raw.parquet", index=False)
        print(f"  kpi_raw.parquet   → {len(raw_df)} rows")

    # ── Forecast (optional) ──────────────────────────────────────────────────
    has_forecast    = False
    forecast_horizon = 0
    if forecast_csv:
        fcast_df = pd.read_csv(forecast_csv)
        fcast_df.to_parquet(out / "forecast.parquet", index=False)
        has_forecast     = True
        forecast_horizon = int(fcast_df["horizon"].max())
        print(f"  forecast.parquet  → {len(fcast_df)} rows (horizon 1..{forecast_horizon})")

    # ── Influence Scores ─────────────────────────────────────────────────────
    print("  computing influence scores …")
    is_df = compute_influence_scores(sym_df, kpi_cols)
    is_df.to_parquet(out / "influence_score.parquet", index=False)
    print(f"  influence_score.parquet → {len(is_df)} rows")

    # ── Mutual Information ───────────────────────────────────────────────────
    mi = compute_mutual_information(sym_df, kpi_cols)
    (out / "mi.json").write_text(json.dumps(mi, indent=2))
    print(f"  mi.json → {mi}")

    # ── Refinement Events ────────────────────────────────────────────────────
    ref_df = extract_refinement_events(sym_df)
    ref_df.to_parquet(out / "refinement_events.parquet", index=False)
    print(f"  refinement_events.parquet → {len(ref_df)} override events")

    # ── Meta ─────────────────────────────────────────────────────────────────
    slices_present = sorted(sym_df["slice_id"].unique().tolist())
    meta = {
        "use_case":          use_case,
        "agent":             agent,
        "run_id":            Path(out_dir).name,
        "kpi_cols":          kpi_cols,
        "action_cols":       action_cols,
        "action_combined_col": "action_combined",
        "slices":            slices_present,
        "slice_names":       {str(s): snames.get(s, str(s)) for s in slices_present},
        "max_t":             int(sym_df["timestep"].max()),
        "n_train":           int(sym_df[sym_df["training"] == True]["timestep"].nunique()),
        "has_forecast":      has_forecast,
        "forecast_horizon":  forecast_horizon,
        "has_refinement":    bool(sym_df["able_to_improve"].any()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta.json → max_t={meta['max_t']}, slices={meta['slices']}, "
          f"has_forecast={has_forecast}")
    print(f"Done → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SIA dashboard fixture")
    parser.add_argument("--symbolic",  required=True)
    parser.add_argument("--kpi-raw",   default=None)
    parser.add_argument("--forecast",  default=None, help="Path to forecast CSV (optional)")
    parser.add_argument("--use-case",  required=True, choices=list(_USE_CASE_CONFIGS))
    parser.add_argument("--agent",     required=True)
    parser.add_argument("--out-dir",   required=True)
    args = parser.parse_args()

    print(f"Exporting {args.use_case}/{args.agent} …")
    export_run(
        symbolic_csv = args.symbolic,
        kpi_raw_csv  = args.kpi_raw,
        use_case     = args.use_case,
        agent        = args.agent,
        out_dir      = args.out_dir,
        forecast_csv = args.forecast,
    )


if __name__ == "__main__":
    main()
