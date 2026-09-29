#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="results/latest")
    args = p.parse_args()
    d = Path(args.run_dir)
    df = pd.read_csv(d / "episode_metrics.csv")
    metrics = ["grid_safe_rate", "voltage_violation_rate", "thermal_violation_rate", "mean_grid_import_mw", "renewable_curtailment_mwh_proxy", "intervention_rate", "mean_action_distance"]
    summary = df.groupby(["controller", "scenario"])[metrics].agg(["mean", "std"]).reset_index()
    summary.to_csv(d / "summary_by_controller_scenario.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
