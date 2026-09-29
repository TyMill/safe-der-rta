#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def q(x: pd.Series, p: float) -> float:
    return float(x.quantile(p)) if len(x) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize reserve-only vs RTA ablation and timing.")
    ap.add_argument("--run-dir", default="results/latest")
    args = ap.parse_args()
    run = Path(args.run_dir)
    steps = pd.read_csv(run / "step_metrics.csv")
    eps = pd.read_csv(run / "episode_metrics.csv")

    safety = (
        eps.groupby(["controller", "scenario"], as_index=False)
        .agg(
            n=("seed", "count"),
            safe_rate_mean=("grid_safe_rate", "mean"),
            safe_rate_sd=("grid_safe_rate", "std"),
            reject_rate_mean=("reject_rate", "mean"),
            intervention_rate_mean=("runtime_intervention_rate", "mean"),
        )
    )
    safety.to_csv(run / "ablation_safety_summary.csv", index=False)

    # Paired seed-level safety gain from adding RTA to the same reserve-aware policy family.
    wide = eps.pivot_table(index=["scenario", "seed"], columns="controller", values="grid_safe_rate")
    if {"reserve_only", "assured"}.issubset(wide.columns):
        paired = wide.dropna(subset=["reserve_only", "assured"]).copy()
        paired["rta_gain_pp"] = 100.0 * (paired["assured"] - paired["reserve_only"])
        gain = paired.reset_index().groupby("scenario", as_index=False).agg(
            n=("seed", "count"),
            reserve_only_mean=("reserve_only", "mean"),
            assured_mean=("assured", "mean"),
            rta_gain_pp_mean=("rta_gain_pp", "mean"),
            rta_gain_pp_sd=("rta_gain_pp", "std"),
            rta_gain_pp_min=("rta_gain_pp", "min"),
            rta_gain_pp_max=("rta_gain_pp", "max"),
        )
        gain.to_csv(run / "ablation_paired_gain.csv", index=False)

    records = []
    for keys, g in steps.groupby(["controller", "scenario", "assurance_decision"], dropna=False):
        controller, scenario, decision = keys
        records.append({
            "controller": controller,
            "scenario": scenario,
            "decision": decision,
            "n_steps": len(g),
            "proposal_ms_median": float(g["proposal_latency_ms"].median()),
            "proposal_ms_p95": q(g["proposal_latency_ms"], 0.95),
            "safety_layer_ms_median": float(g["safety_layer_latency_ms"].median()),
            "safety_layer_ms_p95": q(g["safety_layer_latency_ms"], 0.95),
            "safety_layer_ms_p99": q(g["safety_layer_latency_ms"], 0.99),
            "compute_ms_median": float(g["measured_compute_latency_ms"].median()),
            "compute_ms_p95": q(g["measured_compute_latency_ms"], 0.95),
            "compute_ms_p99": q(g["measured_compute_latency_ms"], 0.99),
            "pf_calls_safety_mean": float(g["pf_calls_safety_layer"].mean()),
            "pf_calls_safety_p95": q(g["pf_calls_safety_layer"], 0.95),
            "pf_calls_total_mean": float(g["pf_calls_total"].mean()),
            "pf_calls_total_max": int(g["pf_calls_total"].max()),
        })
    pd.DataFrame(records).to_csv(run / "latency_pf_summary.csv", index=False)

    print("Wrote:")
    for name in ["ablation_safety_summary.csv", "ablation_paired_gain.csv", "latency_pf_summary.csv"]:
        path = run / name
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()
