from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def q95(x): return x.quantile(0.95)
def q99(x): return x.quantile(0.99)

p = argparse.ArgumentParser()
p.add_argument("--run-dir", default="results/latest")
a = p.parse_args()
run = Path(a.run_dir)
steps = pd.read_csv(run / "step_metrics.csv")
eps = pd.read_csv(run / "episode_metrics.csv")

summary = eps.groupby("scenario", as_index=False).agg(
    seeds=("seed", "nunique"),
    mean_grid_safe=("grid_safe_rate", "mean"),
    sd_grid_safe=("grid_safe_rate", "std"),
    mean_import_mw=("mean_grid_import_mw", "mean"),
    mean_losses_mwh=("total_losses_mwh", "mean"),
    mean_curtailment_mwh=("renewable_curtailment_mwh", "mean"),
    median_compute_ms=("median_measured_compute_latency_ms", "median"),
    p95_compute_ms=("p95_measured_compute_latency_ms", "median"),
    mean_pf_calls=("mean_pf_calls_total", "mean"),
)
summary.to_csv(run / "hard_constrained_summary.csv", index=False)

status = (steps.groupby(["scenario", "solver_status"]).size().rename("steps").reset_index())
status["share"] = status["steps"] / status.groupby("scenario")["steps"].transform("sum")
status.to_csv(run / "hard_constrained_solver_status.csv", index=False)

lat = steps.groupby("scenario", as_index=False).agg(
    median_ms=("measured_compute_latency_ms", "median"),
    p95_ms=("measured_compute_latency_ms", q95),
    p99_ms=("measured_compute_latency_ms", q99),
    max_ms=("measured_compute_latency_ms", "max"),
    median_pf_calls=("pf_calls_total", "median"),
    p95_pf_calls=("pf_calls_total", q95),
    max_pf_calls=("pf_calls_total", "max"),
    solver_success_rate=("solver_success", "mean"),
)
lat.to_csv(run / "hard_constrained_latency_pf.csv", index=False)

print("\nSafety/operations:\n", summary.to_string(index=False))
print("\nSolver status:\n", status.to_string(index=False))
print("\nLatency/PF:\n", lat.to_string(index=False))
