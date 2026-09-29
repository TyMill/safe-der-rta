from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import yaml

from safe_der.agents.policies import resource_bounds
from safe_der.grid.model import DistributionGrid
from safe_der.scenarios.synthetic import generate_profile
from safe_der.types import DispatchAction, ResourceState


def _candidate_actions(cfg: dict, state: ResourceState, row: pd.Series) -> list[DispatchAction]:
    """Coarse state-aware feasibility oracle.

    The oracle is intentionally independent of the runtime-assurance search logic.
    It spans physical resource bounds at the *actual pre-decision SoC* and uses a
    deterministic lattice over BESS, EV, flexible-load relief and curtailment.
    It is an analysis instrument, not a controller and not a source of online actions.
    """
    b = resource_bounds(cfg, state, float(row["ev_connected"]))
    bvals = np.linspace(b["battery_mw"][0], b["battery_mw"][1], 5)
    evals = np.linspace(b["ev_mw"][0], b["ev_mw"][1], 5)
    fvals = np.linspace(0.0, b["flex_load_mw"][1], 3)
    pvals = (0.0, 0.5 * float(row["pv_mw"]), float(row["pv_mw"]))
    wvals = (0.0, 0.5 * float(row["wind_mw"]), float(row["wind_mw"]))
    out: list[DispatchAction] = []
    for bp in bvals:
        for ep in evals:
            for fp in fvals:
                for pc in pvals:
                    for wc in wvals:
                        out.append(DispatchAction(float(bp), float(ep), float(fp), float(pc), float(wc)))
    return out


def _load_cfg(run_dir: Path) -> dict:
    with open(run_dir / "config_resolved.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_pre_state(group: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Support both v0.6 outputs and older v0.5 result folders."""
    g = group.sort_values("step").copy()
    if "battery_soc_before" in g.columns and "ev_soc_before" in g.columns:
        return g
    a = cfg["assets"]
    b_before = [float(a["battery_soc_initial"])]
    e_before = [float(a["ev_soc_initial"])]
    if len(g) > 1:
        b_before.extend(g["battery_soc"].iloc[:-1].astype(float).tolist())
        e_before.extend(g["ev_soc"].iloc[:-1].astype(float).tolist())
    g["battery_soc_before"] = b_before
    g["ev_soc_before"] = e_before
    return g


def analyze(run_dir: Path, controller: str = "assured") -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = _load_cfg(run_dir)
    steps = pd.read_csv(run_dir / "step_metrics.csv")
    steps = steps.loc[steps["controller"] == controller].copy()
    if steps.empty:
        raise ValueError(f"No rows for controller={controller!r}")

    grid = DistributionGrid(cfg)
    detailed: list[dict] = []

    for (scenario, seed), raw_group in steps.groupby(["scenario", "seed"], sort=True):
        group = _ensure_pre_state(raw_group, cfg)
        # Older v0.5 result files did not log ev_connected; regenerate only that
        # exogenous field from the frozen scenario generator when needed.
        if "ev_connected" not in group.columns:
            prof = generate_profile(cfg, str(scenario), int(seed)).set_index("step")
            group["ev_connected"] = group["step"].map(prof["ev_connected"])

        for _, row in group.iterrows():
            if bool(row["grid_safe"]):
                # The actually executed action is already a constructive witness.
                state_feasible = True
                witness = DispatchAction(
                    float(row["battery_mw"]), float(row["ev_mw"]),
                    float(row["flex_load_mw"]), float(row["pv_curtail_mw"]),
                    float(row["wind_curtail_mw"]),
                )
                n_tested = 0
                status = "safe_executed"
            else:
                state = ResourceState(float(row["battery_soc_before"]), float(row["ev_soc_before"]))
                state_feasible = False
                witness = None
                n_tested = 0
                for action in _candidate_actions(cfg, state, row):
                    n_tested += 1
                    ass = grid.assess(float(row["demand_factor"]), float(row["pv_mw"]), float(row["wind_mw"]), action)
                    if ass.safe:
                        state_feasible = True
                        witness = action
                        break
                status = "recoverable_but_missed" if state_feasible else "no_safe_lattice_witness"

            wd = witness.as_dict() if witness is not None else {k: np.nan for k in DispatchAction().as_dict()}
            detailed.append({
                "controller": controller,
                "scenario": scenario,
                "seed": int(seed),
                "step": int(row["step"]),
                "hour": float(row["hour"]),
                "grid_safe": bool(row["grid_safe"]),
                "trajectory_state_feasible": bool(state_feasible),
                "trajectory_status": status,
                "battery_soc_before": float(row["battery_soc_before"]),
                "ev_soc_before": float(row["ev_soc_before"]),
                "candidate_actions_tested": int(n_tested),
                **{f"witness_{k}": v for k, v in wd.items()},
            })

    detail = pd.DataFrame(detailed)
    summary_rows: list[dict] = []
    for (scenario, seed), g in detail.groupby(["scenario", "seed"], sort=True):
        unsafe = ~g["grid_safe"]
        summary_rows.append({
            "controller": controller,
            "scenario": scenario,
            "seed": int(seed),
            "steps": int(len(g)),
            "grid_safe_rate": float(g["grid_safe"].mean()),
            "trajectory_state_feasible_rate": float(g["trajectory_state_feasible"].mean()),
            "unsafe_steps": int(unsafe.sum()),
            "no_safe_lattice_witness_steps": int((g["trajectory_status"] == "no_safe_lattice_witness").sum()),
            "recoverable_but_missed_steps": int((g["trajectory_status"] == "recoverable_but_missed").sum()),
            "unsafe_explained_by_state_infeasibility_rate": (
                float((g["trajectory_status"] == "no_safe_lattice_witness").sum() / unsafe.sum()) if unsafe.sum() else float("nan")
            ),
        })
    summary = pd.DataFrame(summary_rows)
    return detail, summary


def main() -> None:
    p = argparse.ArgumentParser(description="State-aware post-hoc AC feasibility analysis.")
    p.add_argument("--run-dir", default="results/latest")
    p.add_argument("--controller", default="assured")
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    detail, summary = analyze(run_dir, args.controller)
    detail.to_csv(run_dir / "trajectory_feasibility.csv", index=False)
    summary.to_csv(run_dir / "trajectory_feasibility_summary.csv", index=False)

    by_scenario = summary.groupby("scenario").agg(
        mean_grid_safe_rate=("grid_safe_rate", "mean"),
        mean_trajectory_state_feasible_rate=("trajectory_state_feasible_rate", "mean"),
        total_unsafe_steps=("unsafe_steps", "sum"),
        total_no_safe_lattice_witness_steps=("no_safe_lattice_witness_steps", "sum"),
        total_recoverable_but_missed_steps=("recoverable_but_missed_steps", "sum"),
    )
    print(by_scenario.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {run_dir / 'trajectory_feasibility.csv'}")
    print(f"Saved: {run_dir / 'trajectory_feasibility_summary.csv'}")


if __name__ == "__main__":
    main()
