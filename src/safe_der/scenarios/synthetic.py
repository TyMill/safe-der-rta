from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = {"normal", "solar_surge", "renewable_drop", "demand_spike", "ev_coincidence", "compound", "extreme_compound"}


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _base_profiles(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 24 h backbone independent of experiment length."""
    demand = 0.58 + 0.16 * np.exp(-((hour - 8.0) / 2.4) ** 2) + 0.28 * np.exp(-((hour - 19.0) / 3.0) ** 2)
    solar = np.sin(np.pi * np.clip((hour - 6.0) / 12.0, 0, 1)) ** 1.8
    # Wind is tied to clock hour, not t/n, so smoke and full-day runs share the same physics.
    wind = 0.48 + 0.16 * np.sin(2 * np.pi * hour / 24.0 + 0.7)
    ev_connected = 0.15 + 0.75 * np.exp(-((hour - 20.0) / 3.8) ** 2)
    price = 0.45 + 0.20 * np.exp(-((hour - 19.0) / 3.2) ** 2) + 0.08 * np.exp(-((hour - 8.0) / 2.5) ** 2)
    return demand, solar, wind, ev_connected, price


def deterministic_normal_profile(cfg: dict, horizon_steps: int | None = None) -> pd.DataFrame:
    """Noise-free nominal profile used only for benchmark thermal-rating calibration."""
    n = int(horizon_steps or round(24 * 60 / int(cfg["experiment"]["step_minutes"])))
    dt_min = int(cfg["experiment"]["step_minutes"])
    t = np.arange(n)
    hour = (t * dt_min / 60.0) % 24.0
    demand, solar, wind, ev_connected, price = _base_profiles(hour)
    a = cfg["assets"]
    return pd.DataFrame({
        "step": t,
        "hour": hour,
        "demand_factor": np.clip(demand, 0.25, 1.6),
        "pv_mw": _clip01(solar) * float(a["pv_scale_mw"]),
        "wind_mw": _clip01(wind) * float(a["wind_scale_mw"]),
        "price": price,
        "ev_connected": _clip01(ev_connected),
    })


def generate_profile(cfg: dict, scenario: str, seed: int) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    n = int(cfg["experiment"]["horizon_steps"])
    dt_min = int(cfg["experiment"]["step_minutes"])
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    hour = (t * dt_min / 60.0) % 24.0

    demand, solar, wind, ev_connected, price = _base_profiles(hour)
    demand = demand * (1.0 + rng.normal(0.0, cfg["uncertainty"]["demand_noise_std"], n))
    solar = solar * (1.0 + rng.normal(0.0, cfg["uncertainty"]["renewable_noise_std"], n))
    wind = wind + rng.normal(0.0, cfg["uncertainty"]["renewable_noise_std"], n)

    # Fixed clock-time disturbance window: 10:00-15:00 local simulation time.
    event = (hour >= 10.0) & (hour < 15.0)
    if scenario == "solar_surge":
        solar[event] *= 1.55
    elif scenario == "renewable_drop":
        solar[event] *= 0.25
        wind[event] *= 0.45
    elif scenario == "demand_spike":
        demand[event] *= 1.45
    elif scenario == "ev_coincidence":
        ev_connected[event] = np.clip(ev_connected[event] + 0.65, 0, 1)
        demand[event] *= 1.20
    elif scenario == "compound":
        # Recoverable compound stress: simultaneous renewable shortfall, elevated
        # demand, and coincident EV availability. Severity is intentionally below
        # the physical infeasibility boundary so runtime assurance can be judged on
        # its ability to find an existing safe action.
        solar[event] *= 0.55
        wind[event] *= 0.70
        demand[event] *= 1.25
        ev_connected[event] = 1.0
    elif scenario == "extreme_compound":
        # Deliberately severe boundary test. Safe operation may be physically
        # infeasible with the configured DER fleet; correct assurance behaviour is
        # then explicit reject/best-effort rather than a false safety claim.
        solar[event] *= 0.30
        wind[event] *= 0.50
        demand[event] *= 1.48
        ev_connected[event] = 1.0

    a = cfg["assets"]
    return pd.DataFrame({
        "step": t,
        "hour": hour,
        "demand_factor": np.clip(demand, 0.25, 1.6),
        "pv_mw": _clip01(solar) * float(a["pv_scale_mw"]),
        "wind_mw": _clip01(wind) * float(a["wind_scale_mw"]),
        "price": price,
        "ev_connected": _clip01(ev_connected),
    })


def forecast_row(row: pd.Series, cfg: dict, rng: np.random.Generator) -> pd.Series:
    out = row.copy()
    s = float(cfg["uncertainty"]["forecast_error_std"])
    for col in ["demand_factor", "pv_mw", "wind_mw"]:
        out[col] = max(0.0, float(row[col]) * (1.0 + rng.normal(0.0, s)))
    return out
