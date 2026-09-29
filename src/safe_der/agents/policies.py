from __future__ import annotations

from safe_der.types import DispatchAction, ResourceState


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def resource_bounds(cfg: dict, state: ResourceState, ev_connected: float) -> dict[str, tuple[float, float]]:
    a = cfg["assets"]
    dt = cfg["experiment"]["step_minutes"] / 60.0
    eta = float(a["battery_efficiency"])
    bpow = float(a["battery_power_mw"])
    benergy = float(a["battery_energy_mwh"])
    bmin, bmax = float(a["battery_soc_min"]), float(a["battery_soc_max"])
    max_discharge_soc = max(0.0, (state.battery_soc - bmin) * benergy * eta / dt)
    max_charge_soc = max(0.0, (bmax - state.battery_soc) * benergy / eta / dt)

    epow = float(a["ev_power_mw"]) * float(ev_connected)
    # EV SoC kept in broad [0.2, 0.9] envelope for synthetic benchmark.
    eenergy = float(a["ev_energy_mwh"])
    max_ev_dis = max(0.0, (state.ev_soc - 0.20) * eenergy * eta / dt)
    max_ev_ch = max(0.0, (0.90 - state.ev_soc) * eenergy / eta / dt)

    return {
        "battery_mw": (-min(bpow, max_charge_soc), min(bpow, max_discharge_soc)),
        "ev_mw": (-min(epow, max_ev_ch), min(epow, max_ev_dis)),
        "flex_load_mw": (0.0, float(a["flexible_load_mw"])),
    }


def operational_resource_bounds(cfg: dict, state: ResourceState, ev_connected: float) -> dict[str, tuple[float, float]]:
    """Resource bounds used by the assured proposal policy.

    These bounds preserve configurable battery/EV reserves during routine dispatch.
    They are intentionally stricter than the physical bounds returned by
    :func:`resource_bounds`; runtime assurance may still consume the reserve when
    required to restore electrical safety.
    """
    physical = resource_bounds(cfg, state, ev_connected)
    ass_cfg = cfg.get("assurance", {})
    a = cfg["assets"]
    dt = float(cfg["experiment"]["step_minutes"]) / 60.0
    eta = float(a["battery_efficiency"])

    b_res = float(ass_cfg.get("battery_reserve_soc", a["battery_soc_min"]))
    e_res = float(ass_cfg.get("ev_reserve_soc", 0.20))
    b_cap = float(a["battery_energy_mwh"])
    e_cap = float(a["ev_energy_mwh"])

    max_b_dis_reserve = max(0.0, (state.battery_soc - b_res) * b_cap * eta / dt)
    max_e_dis_reserve = max(0.0, (state.ev_soc - e_res) * e_cap * eta / dt)

    return {
        "battery_mw": (physical["battery_mw"][0], min(physical["battery_mw"][1], max_b_dis_reserve)),
        "ev_mw": (physical["ev_mw"][0], min(physical["ev_mw"][1], max_e_dis_reserve)),
        "flex_load_mw": physical["flex_load_mw"],
    }


def action_within_operational_bounds(action: DispatchAction, cfg: dict, state: ResourceState, row, tol: float = 1e-9) -> bool:
    """Return True when a proposal respects routine resource/reserve constraints."""
    b = operational_resource_bounds(cfg, state, float(row["ev_connected"]))
    values = {
        "battery_mw": action.battery_mw,
        "ev_mw": action.ev_mw,
        "flex_load_mw": action.flex_load_mw,
    }
    for key, value in values.items():
        lo, hi = b[key]
        if value < lo - tol or value > hi + tol:
            return False
    if action.pv_curtail_mw < -tol or action.pv_curtail_mw > float(row["pv_mw"]) + tol:
        return False
    if action.wind_curtail_mw < -tol or action.wind_curtail_mw > float(row["wind_mw"]) + tol:
        return False
    return True


def clip_action(action: DispatchAction, cfg: dict, state: ResourceState, row) -> DispatchAction:
    b = resource_bounds(cfg, state, float(row["ev_connected"]))
    return DispatchAction(
        battery_mw=clamp(action.battery_mw, *b["battery_mw"]),
        ev_mw=clamp(action.ev_mw, *b["ev_mw"]),
        flex_load_mw=clamp(action.flex_load_mw, *b["flex_load_mw"]),
        pv_curtail_mw=clamp(action.pv_curtail_mw, 0.0, float(row["pv_mw"])),
        wind_curtail_mw=clamp(action.wind_curtail_mw, 0.0, float(row["wind_mw"])),
    )


def update_resource_state(state: ResourceState, action: DispatchAction, cfg: dict) -> ResourceState:
    a = cfg["assets"]
    dt = cfg["experiment"]["step_minutes"] / 60.0
    eta = float(a["battery_efficiency"])
    def step_soc(soc: float, p: float, cap: float) -> float:
        # Positive p = discharge. Charging incurs efficiency; discharge divides by efficiency.
        delta = (-p * dt / eta if p > 0 else -p * dt * eta) / cap
        return float(max(0.0, min(1.0, soc + delta)))
    return ResourceState(
        battery_soc=step_soc(state.battery_soc, action.battery_mw, float(a["battery_energy_mwh"])),
        ev_soc=step_soc(state.ev_soc, action.ev_mw, float(a["ev_energy_mwh"])),
        previous_action=action,
    )
