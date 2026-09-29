from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from safe_der.agents.policies import action_within_operational_bounds, update_resource_state
from safe_der.assurance.runtime import RuntimeAssurance
from safe_der.coordination.controllers import propose
from safe_der.coordination.hard_constrained import solve_hard_constrained
from safe_der.grid.model import DistributionGrid
from safe_der.metrics.core import aggregate_episode
from safe_der.scenarios.synthetic import forecast_row, generate_profile
from safe_der.types import AssuranceResult, DispatchAction, ResourceState


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_episode(cfg: dict, controller: str, scenario: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = DistributionGrid(cfg)
    assurance = RuntimeAssurance(cfg, grid)
    profile = generate_profile(cfg, scenario, seed)
    rng = np.random.default_rng(seed + 99991)
    a = cfg["assets"]
    state = ResourceState(float(a["battery_soc_initial"]), float(a["ev_soc_initial"]))
    rows, interventions = [], []

    for _, actual in profile.iterrows():
        # Preserve the causal pre-decision resource state for post-hoc reachability
        # analysis. These fields are observational only and do not affect control.
        battery_soc_before = float(state.battery_soc)
        ev_soc_before = float(state.ev_soc)
        forecast = forecast_row(actual, cfg, rng)

        # Instrument proposal generation separately from the runtime safety layer.
        # Timing is observational only; it does not affect controller decisions.
        pf0 = int(grid.assess_calls)
        t0 = perf_counter()
        hard_result = None
        if controller == "hard_constrained":
            # Conventional online AC-constrained optimization sees the same current
            # measured exogenous state available to the runtime assurance layer.
            hard_result = solve_hard_constrained(actual, cfg, state, grid)
            proposal = hard_result.executed
        else:
            proposal = propose(controller, forecast, cfg, state, grid)
        proposal_latency_ms = (perf_counter() - t0) * 1000.0
        pf1 = int(grid.assess_calls)
        pf_calls_proposal = pf1 - pf0

        if rng.random() < float(cfg["uncertainty"].get("agent_dropout_prob", 0.0)):
            proposal = DispatchAction()

        # For hard_constrained, physical rather than reserve-preserving bounds define
        # admissibility. Keep this field for schema compatibility but do not use it to
        # interpret the conventional optimizer's success.
        proposal_operational_feasible = (
            True if controller == "hard_constrained"
            else action_within_operational_bounds(proposal, cfg, state, actual)
        )

        safety_layer_latency_ms = 0.0
        grid_evaluation_latency_ms = 0.0
        pf_calls_safety_layer = 0
        pf_calls_evaluation = 0
        solver_success = None
        solver_status = "not_applicable"
        solver_message = ""
        solver_iterations = 0
        solver_objective = float("nan")
        solver_feasible_points_seen = 0
        solver_evaluated_points = 0

        if controller == "hard_constrained":
            assert hard_result is not None
            executed = hard_result.executed
            assessment = hard_result.assessment
            decision = "approve" if assessment.safe else "reject"
            distance = 0.0
            repair_iterations = 0
            solver_success = hard_result.solver_success
            solver_status = hard_result.solver_status
            solver_message = hard_result.solver_message
            solver_iterations = hard_result.solver_iterations
            solver_objective = hard_result.objective_value
            solver_feasible_points_seen = hard_result.feasible_points_seen
            solver_evaluated_points = hard_result.evaluated_points
        elif controller == "assured":
            pf2 = int(grid.assess_calls)
            t1 = perf_counter()
            result = assurance.evaluate(proposal, actual, state)
            safety_layer_latency_ms = (perf_counter() - t1) * 1000.0
            pf3 = int(grid.assess_calls)
            pf_calls_safety_layer = pf3 - pf2
            executed = result.executed
            assessment = result.assessment
            decision = result.decision
            distance = result.action_distance
            repair_iterations = result.repair_iterations
        else:
            executed = proposal
            pf2 = int(grid.assess_calls)
            t1 = perf_counter()
            assessment = grid.assess(float(actual["demand_factor"]), float(actual["pv_mw"]), float(actual["wind_mw"]), executed)
            grid_evaluation_latency_ms = (perf_counter() - t1) * 1000.0
            pf3 = int(grid.assess_calls)
            pf_calls_evaluation = pf3 - pf2
            decision = "approve"
            distance = 0.0
            repair_iterations = 0

        decision_latency_ms = proposal_latency_ms + safety_layer_latency_ms
        measured_compute_latency_ms = proposal_latency_ms + safety_layer_latency_ms + grid_evaluation_latency_ms
        pf_calls_total = pf_calls_proposal + pf_calls_safety_layer + pf_calls_evaluation

        state = update_resource_state(state, executed, cfg)
        rec = {
            "controller": controller, "scenario": scenario, "seed": seed, "step": int(actual["step"]), "hour": float(actual["hour"]),
            "demand_factor": float(actual["demand_factor"]), "pv_mw": float(actual["pv_mw"]), "wind_mw": float(actual["wind_mw"]), "price": float(actual["price"]),
            "ev_connected": float(actual["ev_connected"]),
            "battery_soc_before": battery_soc_before, "ev_soc_before": ev_soc_before,
            **executed.as_dict(),
            **{f"proposed_{k}": v for k, v in proposal.as_dict().items()},
            "proposal_operational_feasible": proposal_operational_feasible,
            "runtime_intervened": bool(controller == "assured" and decision != "approve"),
            "battery_soc": state.battery_soc, "ev_soc": state.ev_soc,
            "grid_safe": assessment.safe, "converged": assessment.converged,
            "min_voltage_pu": assessment.min_voltage_pu, "max_voltage_pu": assessment.max_voltage_pu,
            "max_line_loading_pct": assessment.max_line_loading_pct, "grid_import_mw": assessment.grid_import_mw, "losses_mw": assessment.losses_mw,
            "voltage_violation": assessment.voltage_violation, "thermal_violation": assessment.thermal_violation,
            "assurance_decision": decision, "action_distance": distance, "repair_iterations": repair_iterations,
            "proposal_latency_ms": proposal_latency_ms,
            "safety_layer_latency_ms": safety_layer_latency_ms,
            "grid_evaluation_latency_ms": grid_evaluation_latency_ms,
            "decision_latency_ms": decision_latency_ms,
            "measured_compute_latency_ms": measured_compute_latency_ms,
            "pf_calls_proposal": pf_calls_proposal,
            "pf_calls_safety_layer": pf_calls_safety_layer,
            "pf_calls_evaluation": pf_calls_evaluation,
            "pf_calls_total": pf_calls_total,
            "solver_success": solver_success,
            "solver_status": solver_status,
            "solver_message": solver_message,
            "solver_iterations": solver_iterations,
            "solver_objective": solver_objective,
            "solver_feasible_points_seen": solver_feasible_points_seen,
            "solver_evaluated_points": solver_evaluated_points,
        }
        rows.append(rec)
        if controller == "assured" and decision != "approve":
            interventions.append({
                "controller": controller, "scenario": scenario, "seed": seed, "step": int(actual["step"]),
                "decision": decision, "action_distance": distance, "repair_iterations": repair_iterations,
                **{f"proposed_{k}": v for k, v in proposal.as_dict().items()},
                **{f"executed_{k}": v for k, v in executed.as_dict().items()},
            })
    return pd.DataFrame(rows), pd.DataFrame(interventions)


def make_run_dir(base: Path, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = base / f"{stamp}_{name}"
    out.mkdir(parents=True, exist_ok=False)
    latest = base / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out.name, target_is_directory=True)
    except OSError:
        pass
    return out


def run_campaign(config_path: str | Path, output_root: str | Path = "results") -> Path:
    cfg = load_config(config_path)
    out = make_run_dir(Path(output_root), cfg["experiment"]["name"])
    with open(out / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    step_frames, int_frames, episodes = [], [], []
    jobs = [(c, s, int(seed)) for c in cfg["experiment"]["controllers"] for s in cfg["experiment"]["scenarios"] for seed in cfg["experiment"]["seeds"]]
    for controller, scenario, seed in tqdm(jobs, desc="experiments"):
        df, ints = run_episode(cfg, controller, scenario, seed)
        step_frames.append(df)
        if not ints.empty:
            int_frames.append(ints)
        agg = aggregate_episode(df, cfg)
        episodes.append({"controller": controller, "scenario": scenario, "seed": seed, **agg})

    steps = pd.concat(step_frames, ignore_index=True)
    eps = pd.DataFrame(episodes)
    ints = pd.concat(int_frames, ignore_index=True) if int_frames else pd.DataFrame()
    steps.to_csv(out / "step_metrics.csv", index=False)
    eps.to_csv(out / "episode_metrics.csv", index=False)
    ints.to_csv(out / "interventions.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "config": str(config_path), "episodes": len(eps), "steps": len(steps),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--output-root", default="results")
    args = p.parse_args()
    out = run_campaign(args.config, args.output_root)
    print(out)


if __name__ == "__main__":
    main()
