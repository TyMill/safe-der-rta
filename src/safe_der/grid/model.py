from __future__ import annotations

import copy
import numpy as np
import pandapower as pp
import pandapower.networks as pn

from safe_der.types import DispatchAction, GridAssessment
from safe_der.scenarios.synthetic import deterministic_normal_profile


class DistributionGrid:
    """IEEE-33 AC power-flow wrapper with synthetic DER placement.

    For the synthetic benchmark, thermal ampacities can be calibrated against a
    deterministic, noise-free 24 h normal operating envelope that includes the
    configured DER fleet. This avoids defining ratings from a no-DER snapshot
    that becomes unrealistically restrictive under ordinary reverse power flow.
    """

    def __init__(self, cfg: dict):
        if cfg["network"]["case"] != "ieee33":
            raise ValueError("Current benchmark release supports network.case=ieee33")
        self.cfg = cfg
        self.base = pn.case33bw()
        self.base_load_p = self.base.load.p_mw.to_numpy(copy=True)
        self.base_load_q = self.base.load.q_mvar.to_numpy(copy=True)
        self.pv_bus = 17
        self.wind_bus = 30
        self.battery_bus = 13
        self.ev_bus = 24
        self.flex_bus = 7
        self.assess_calls = 0
        self._calibrate_thermal_ratings_if_requested()

    def _make_net(self, demand_factor: float, pv_mw: float, wind_mw: float, action: DispatchAction):
        net = copy.deepcopy(self.base)
        net.load["p_mw"] = self.base_load_p * float(demand_factor)
        net.load["q_mvar"] = self.base_load_q * float(demand_factor)

        flex_idx = int(np.argmin(np.abs(net.load.bus.to_numpy() - self.flex_bus)))
        net.load.loc[flex_idx, "p_mw"] = max(0.0, net.load.loc[flex_idx, "p_mw"] - action.flex_load_mw)

        pp.create_sgen(net, self.pv_bus, p_mw=max(0.0, pv_mw - action.pv_curtail_mw), q_mvar=0.0, name="PV")
        pp.create_sgen(net, self.wind_bus, p_mw=max(0.0, wind_mw - action.wind_curtail_mw), q_mvar=0.0, name="Wind")
        if abs(action.battery_mw) > 1e-12:
            pp.create_sgen(net, self.battery_bus, p_mw=action.battery_mw, q_mvar=0.0, name="BESS")
        if abs(action.ev_mw) > 1e-12:
            pp.create_sgen(net, self.ev_bus, p_mw=action.ev_mw, q_mvar=0.0, name="EV")
        return net

    def _calibrate_thermal_ratings_if_requested(self) -> None:
        ncfg = self.cfg["network"]
        if str(ncfg.get("thermal_rating_mode", "native")).lower() != "calibrated":
            return

        # During calibration the actual current does not depend on max_i_ka, so use
        # a nominal 24 h DER-inclusive envelope and collect per-line current maxima.
        profile = deterministic_normal_profile(self.cfg)
        maxima = np.zeros(len(self.base.line), dtype=float)
        for _, row in profile.iterrows():
            net = self._make_net(float(row["demand_factor"]), float(row["pv_mw"]), float(row["wind_mw"]), DispatchAction())
            pp.runpp(net, algorithm="nr", init="auto", calculate_voltage_angles=False, numba=False)
            i_from = np.abs(net.res_line["i_from_ka"].to_numpy(dtype=float))
            i_to = np.abs(net.res_line["i_to_ka"].to_numpy(dtype=float))
            maxima = np.maximum(maxima, np.maximum(i_from, i_to))

        headroom = float(ncfg.get("thermal_headroom_factor", 1.10))
        floor = float(ncfg.get("thermal_min_ampacity_ka", 0.02))
        self.base.line["max_i_ka"] = np.maximum(maxima * headroom, floor)
        self.base.line["df"] = 1.0
        self.base.line["parallel"] = 1

    def assess(self, demand_factor: float, pv_mw: float, wind_mw: float, action: DispatchAction) -> GridAssessment:
        # Count only online/evaluation calls routed through assess(); thermal-rating
        # calibration uses direct pandapower calls and is intentionally excluded.
        self.assess_calls += 1
        net = self._make_net(demand_factor, pv_mw, wind_mw, action)
        try:
            pp.runpp(net, algorithm="nr", init="auto", calculate_voltage_angles=False, numba=False)
            converged = bool(net.converged)
        except Exception:
            converged = False

        if not converged:
            return GridAssessment(False, 0.0, 2.0, 1e6, 1e6, 1e6, 1.0, 1e6)

        vmin = float(net.res_bus.vm_pu.min())
        vmax = float(net.res_bus.vm_pu.max())
        line = float(net.res_line.loading_percent.max())
        ext = float(net.res_ext_grid.p_mw.sum())
        losses = float(net.res_line.pl_mw.sum())
        ncfg = self.cfg["network"]
        vv = max(0.0, float(ncfg["voltage_min_pu"]) - vmin, vmax - float(ncfg["voltage_max_pu"]))
        tv = max(0.0, line - float(ncfg["line_loading_max_pct"]))
        return GridAssessment(True, vmin, vmax, line, ext, losses, vv, tv)
