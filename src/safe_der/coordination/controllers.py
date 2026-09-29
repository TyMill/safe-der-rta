from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from safe_der.agents.policies import clip_action, operational_resource_bounds, resource_bounds
from safe_der.types import DispatchAction, ResourceState


def passive(*_args, **_kwargs) -> DispatchAction:
    return DispatchAction()


def rule_based(row, cfg: dict, state: ResourceState, *_args) -> DispatchAction:
    # Price/demand aware local heuristic; intentionally grid-unaware baseline.
    demand = float(row["demand_factor"])
    price = float(row["price"])
    renewable = float(row["pv_mw"] + row["wind_mw"])
    if demand > 0.85 or price > 0.58:
        a = DispatchAction(battery_mw=0.55, ev_mw=0.20, flex_load_mw=0.18)
    elif renewable > 2.4:
        a = DispatchAction(battery_mw=-0.55, ev_mw=-0.20)
    else:
        a = DispatchAction()
    return clip_action(a, cfg, state, row)


def _vec_to_action(x: np.ndarray) -> DispatchAction:
    return DispatchAction(battery_mw=float(x[0]), ev_mw=float(x[1]), flex_load_mw=float(x[2]), pv_curtail_mw=float(x[3]), wind_curtail_mw=float(x[4]))


def optimize(row, cfg: dict, state: ResourceState, grid, with_penalty: bool, preserve_reserve: bool = False) -> DispatchAction:
    bounds0 = (
        operational_resource_bounds(cfg, state, float(row["ev_connected"]))
        if preserve_reserve
        else resource_bounds(cfg, state, float(row["ev_connected"]))
    )
    bounds = [
        bounds0["battery_mw"], bounds0["ev_mw"], bounds0["flex_load_mw"],
        (0.0, float(row["pv_mw"])), (0.0, float(row["wind_mw"])),
    ]
    prev = np.array(list(state.previous_action.as_dict().values()), dtype=float)
    w = cfg["objective"]

    def objective(x: np.ndarray) -> float:
        a = _vec_to_action(x)
        # Approximate net import proxy plus optional AC-power-flow penalty.
        base_demand_proxy = 3.7 * float(row["demand_factor"])
        net = base_demand_proxy - (float(row["pv_mw"]) - a.pv_curtail_mw) - (float(row["wind_mw"]) - a.wind_curtail_mw) - a.battery_mw - a.ev_mw - a.flex_load_mw
        import_cost = float(row["price"]) * max(net, 0.0)
        curtail = a.pv_curtail_mw + a.wind_curtail_mw
        cycling = abs(a.battery_mw) + 0.5 * abs(a.ev_mw)
        smooth = float(np.square(x - prev).sum())
        value = w["import_cost_weight"] * import_cost + w["curtailment_weight"] * curtail + w["battery_degradation_weight"] * cycling + w["action_smoothness_weight"] * smooth
        if with_penalty:
            ass = grid.assess(float(row["demand_factor"]), float(row["pv_mw"]), float(row["wind_mw"]), a)
            # Normalize violations before applying weights; raw pu excesses are otherwise
            # so small that the penalty baseline collapses to the unconstrained solution.
            value += w["voltage_penalty_weight"] * (ass.voltage_violation / 0.01) ** 2
            value += w["thermal_penalty_weight"] * (ass.thermal_violation / 10.0) ** 2
            if not ass.converged:
                value += 1e5
        return float(value)

    x0 = np.zeros(5, dtype=float)
    # First solve the cheap grid-unaware economic dispatch.
    if not with_penalty:
        result = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60, "ftol": 1e-8})
        return clip_action(_vec_to_action(result.x if result.success else x0), cfg, state, row)

    # Soft-constrained baseline: obtain an economic proposal without AC penalties, then
    # search a small deterministic candidate set. This bounds the number of AC power-flow
    # calls per step and keeps the campaign computationally tractable.
    def economic_only(x: np.ndarray) -> float:
        a = _vec_to_action(x)
        base_demand_proxy = 3.7 * float(row["demand_factor"])
        net = base_demand_proxy - (float(row["pv_mw"]) - a.pv_curtail_mw) - (float(row["wind_mw"]) - a.wind_curtail_mw) - a.battery_mw - a.ev_mw - a.flex_load_mw
        import_cost = float(row["price"]) * max(net, 0.0)
        curtail = a.pv_curtail_mw + a.wind_curtail_mw
        cycling = abs(a.battery_mw) + 0.5 * abs(a.ev_mw)
        smooth = float(np.square(x - prev).sum())
        return float(w["import_cost_weight"] * import_cost + w["curtailment_weight"] * curtail + w["battery_degradation_weight"] * cycling + w["action_smoothness_weight"] * smooth)

    base = minimize(economic_only, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60, "ftol": 1e-8})
    x_star = base.x if base.success else x0
    candidates: list[np.ndarray] = []
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        candidates.append(alpha * x_star)
    # Add two explicit renewable-curtailment candidates for over-voltage conditions.
    for frac in (0.10, 0.25):
        x = x_star.copy()
        x[3] = frac * float(row["pv_mw"])
        x[4] = frac * float(row["wind_mw"])
        candidates.append(x)
    best = min(candidates, key=objective)
    return clip_action(_vec_to_action(best), cfg, state, row)


def propose(controller: str, row, cfg: dict, state: ResourceState, grid) -> DispatchAction:
    if controller == "passive":
        return passive()
    if controller == "rule_based":
        return rule_based(row, cfg, state)
    if controller == "unconstrained":
        return optimize(row, cfg, state, grid, with_penalty=False, preserve_reserve=False)
    if controller in {"reserve_only", "assured"}:
        # Both reserve_only and assured use the exact same reserve-aware economic
        # proposal policy. The only difference is that assured subsequently passes
        # the proposal through RuntimeAssurance, while reserve_only executes it
        # directly. This provides a clean ablation of the runtime safety layer.
        return optimize(row, cfg, state, grid, with_penalty=False, preserve_reserve=True)
    if controller == "penalty":
        return optimize(row, cfg, state, grid, with_penalty=True, preserve_reserve=False)
    raise ValueError(f"Unknown controller: {controller}")
