#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

KEYS = ["scenario", "seed"]
METRICS = [
    "grid_safe_rate", "voltage_violation_rate", "thermal_violation_rate",
    "worst_min_voltage_pu", "max_line_loading_pct", "mean_grid_import_mw",
    "total_losses_mwh", "renewable_curtailment_mwh", "runtime_intervention_rate",
    "modify_rate", "reject_rate", "mean_action_distance", "mean_battery_soc", "mean_ev_soc",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Check that instrumented assured rerun reproduces frozen v0.6 metrics.")
    ap.add_argument("--reference", required=True, help="Frozen v0.6 episode_metrics.csv")
    ap.add_argument("--run-dir", default="results/latest")
    ap.add_argument("--atol", type=float, default=1e-10)
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)
    new = pd.read_csv(Path(args.run_dir) / "episode_metrics.csv")
    ref = ref[ref.controller == "assured"][KEYS + METRICS].copy()
    new = new[new.controller == "assured"][KEYS + METRICS].copy()
    m = ref.merge(new, on=KEYS, suffixes=("_ref", "_new"), validate="one_to_one")
    if len(m) != len(ref) or len(m) != len(new):
        raise SystemExit(f"Key mismatch: reference={len(ref)}, rerun={len(new)}, matched={len(m)}")

    rows = []
    ok = True
    for metric in METRICS:
        d = (m[f"{metric}_ref"] - m[f"{metric}_new"]).abs()
        max_abs = float(d.max())
        rows.append({"metric": metric, "max_abs_difference": max_abs, "within_tolerance": max_abs <= args.atol})
        ok &= max_abs <= args.atol
    out = Path(args.run_dir) / "assured_reproduction_check.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved: {out}")
    if not ok:
        raise SystemExit("Assured rerun does not reproduce frozen v0.6 within tolerance.")
    print("PASS: instrumented assured rerun reproduces frozen v0.6.")


if __name__ == "__main__":
    main()
