from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from safe_der.agents.policies import clip_action, resource_bounds
from safe_der.types import DispatchAction, GridAssessment, ResourceState


@dataclass
class HardConstrainedResult:
    executed: DispatchAction
    assessment: GridAssessment
    solver_success: bool
    solver_status: str
    solver_message: str
    solver_iterations: int
    objective_value: float
    feasible_points_seen: int
    evaluated_points: int


def _vec_to_action(x: np.ndarray) -> DispatchAction:
    return DispatchAction(
        battery_mw=float(x[0]),
        ev_mw=float(x[1]),
        flex_load_mw=float(x[2]),
        pv_curtail_mw=float(x[3]),
        wind_curtail_mw=float(x[4]),
    )


def _objective(x: np.ndarray, row, cfg: dict, state: ResourceState) -> float:
    a = _vec_to_action(x)
    prev = np.array(list(state.previous_action.as_dict().values()), dtype=float)
    w = cfg["objective"]
    base_demand_proxy = 3.7 * float(row["demand_factor"])
    net = (
        base_demand_proxy
        - (float(row["pv_mw"]) - a.pv_curtail_mw)
        - (float(row["wind_mw"]) - a.wind_curtail_mw)
        - a.battery_mw
        - a.ev_mw
        - a.flex_load_mw
    )
    import_cost = float(row["price"]) * max(net, 0.0)
    curtail = a.pv_curtail_mw + a.wind_curtail_mw
    cycling = abs(a.battery_mw) + 0.5 * abs(a.ev_mw)
    smooth = float(np.square(x - prev).sum())
    return float(
        w["import_cost_weight"] * import_cost
        + w["curtailment_weight"] * curtail
        + w["battery_degradation_weight"] * cycling
        + w["action_smoothness_weight"] * smooth
    )


def _violation_score(ass: GridAssessment) -> float:
    if not ass.converged:
        return 1e12
    return float((ass.voltage_violation / 0.01) ** 2 + (ass.thermal_violation / 10.0) ** 2)


def solve_hard_constrained(row, cfg: dict, state: ResourceState, grid) -> HardConstrainedResult:
    """Direct nonlinear AC-constrained dispatch baseline.

    This baseline is intentionally independent of RuntimeAssurance. It optimizes the
    same economic objective over the full physical actuator bounds while enforcing
    current-step AC voltage and thermal constraints through SLSQP. The optimizer sees
    the current measured exogenous state, matching the information available to the
    runtime assurance layer.

    Failure to find a feasible point is reported as a solver/search outcome only; it
    is never interpreted as proof that the continuous feasible set is empty.
    """
    bounds0 = resource_bounds(cfg, state, float(row["ev_connected"]))
    bounds = [
        bounds0["battery_mw"],
        bounds0["ev_mw"],
        bounds0["flex_load_mw"],
        (0.0, float(row["pv_mw"])),
        (0.0, float(row["wind_mw"])),
    ]

    cache: dict[tuple[float, ...], tuple[GridAssessment, DispatchAction]] = {}
    feasible: list[tuple[float, np.ndarray, GridAssessment]] = []
    all_seen: list[tuple[float, float, np.ndarray, GridAssessment]] = []

    def key(x: np.ndarray) -> tuple[float, ...]:
        # SLSQP often asks for the same point repeatedly while estimating gradients.
        # Rounding only affects memoization, never the point sent to the grid model.
        return tuple(np.round(np.asarray(x, dtype=float), 12))

    def assess_x(x: np.ndarray) -> GridAssessment:
        k = key(x)
        if k not in cache:
            action = clip_action(_vec_to_action(np.asarray(x, dtype=float)), cfg, state, row)
            ass = grid.assess(
                float(row["demand_factor"]), float(row["pv_mw"]), float(row["wind_mw"]), action
            )
            cache[k] = (ass, action)
            obj = _objective(np.array(list(action.as_dict().values()), dtype=float), row, cfg, state)
            violation = _violation_score(ass)
            all_seen.append((violation, obj, np.asarray(x, dtype=float).copy(), ass))
            if ass.safe:
                feasible.append((obj, np.asarray(x, dtype=float).copy(), ass))
        return cache[k][0]

    def constraints(x: np.ndarray) -> np.ndarray:
        ass = assess_x(x)
        if not ass.converged:
            return np.array([-1.0, -1.0, -1.0], dtype=float)
        ncfg = cfg["network"]
        return np.array([
            ass.min_voltage_pu - float(ncfg["voltage_min_pu"]),
            float(ncfg["voltage_max_pu"]) - ass.max_voltage_pu,
            (float(ncfg["line_loading_max_pct"]) - ass.max_line_loading_pct) / 100.0,
        ], dtype=float)

    # Economic solution is a natural start and uses no grid penalty or safety layer.
    xzero = np.zeros(5, dtype=float)
    econ = minimize(
        lambda x: _objective(x, row, cfg, state),
        x0=xzero,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 60, "ftol": 1e-8},
    )
    xecon = np.asarray(econ.x if econ.success else xzero, dtype=float)

    # Deterministic initialization set makes the conventional optimizer robust without
    # reusing the RuntimeAssurance recovery library. The best feasible seed (if any)
    # is preferred; otherwise the least-violating seed initializes SLSQP.
    starts: list[np.ndarray] = [xecon, xzero]
    b_lo, b_hi = bounds0["battery_mw"]
    e_lo, e_hi = bounds0["ev_mw"]
    f_hi = bounds0["flex_load_mw"][1]
    pv, wind = float(row["pv_mw"]), float(row["wind_mw"])
    starts.extend([
        np.array([b_hi, e_hi, f_hi, 0.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 0.0, pv, wind], dtype=float),
        np.array([b_lo, e_lo, 0.0, 0.0, 0.0], dtype=float),
        np.array([b_lo, e_lo, 0.0, pv, wind], dtype=float),
    ])
    for x in starts:
        assess_x(x)

    feasible_starts = [item for item in feasible]
    if feasible_starts:
        x0 = min(feasible_starts, key=lambda z: z[0])[1]
    else:
        x0 = min(all_seen, key=lambda z: (z[0], z[1]))[2]

    hcfg: dict[str, Any] = cfg.get("hard_constrained", {})
    result = minimize(
        lambda x: _objective(x, row, cfg, state),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraints}],
        options={
            "maxiter": int(hcfg.get("maxiter", 80)),
            "ftol": float(hcfg.get("ftol", 1e-8)),
            "disp": False,
        },
    )

    x_result = np.asarray(result.x, dtype=float)
    result_ass = assess_x(x_result)
    result_obj = _objective(x_result, row, cfg, state)

    if result.success and result_ass.safe:
        chosen_x, chosen_ass = x_result, result_ass
        status = "success_feasible"
        solver_success = True
    elif feasible:
        _, chosen_x, chosen_ass = min(feasible, key=lambda z: z[0])
        status = "recovered_feasible"
        solver_success = False
    else:
        # Explicit best-effort fallback. This is a numerical-search failure category,
        # not a claim that the physical continuous feasible set is empty.
        _, _, chosen_x, chosen_ass = min(all_seen, key=lambda z: (z[0], z[1]))
        status = "best_effort_no_feasible_found"
        solver_success = False

    chosen_action = clip_action(_vec_to_action(chosen_x), cfg, state, row)
    chosen_obj = _objective(np.array(list(chosen_action.as_dict().values()), dtype=float), row, cfg, state)
    return HardConstrainedResult(
        executed=chosen_action,
        assessment=chosen_ass,
        solver_success=solver_success,
        solver_status=status,
        solver_message=str(result.message),
        solver_iterations=int(getattr(result, "nit", 0)),
        objective_value=float(chosen_obj),
        feasible_points_seen=len(feasible),
        evaluated_points=len(cache),
    )
