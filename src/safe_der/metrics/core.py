from __future__ import annotations

import pandas as pd


def aggregate_episode(df: pd.DataFrame, cfg: dict) -> dict[str, float]:
    dt_h = float(cfg["experiment"]["step_minutes"]) / 60.0
    return {
        "steps": len(df),
        "grid_safe_rate": float(df["grid_safe"].mean()),
        "voltage_violation_rate": float((df["voltage_violation"] > 1e-9).mean()),
        "thermal_violation_rate": float((df["thermal_violation"] > 1e-9).mean()),
        "mean_min_voltage_pu": float(df["min_voltage_pu"].mean()),
        "worst_min_voltage_pu": float(df["min_voltage_pu"].min()),
        "max_line_loading_pct": float(df["max_line_loading_pct"].max()),
        "mean_grid_import_mw": float(df["grid_import_mw"].mean()),
        "total_losses_mwh": float(df["losses_mw"].sum() * dt_h),
        "renewable_curtailment_mwh": float((df["pv_curtail_mw"] + df["wind_curtail_mw"]).sum() * dt_h),
        "proposal_feasible_rate": float(df["proposal_operational_feasible"].mean()),
        "runtime_intervention_rate": float(df["runtime_intervened"].mean()),
        # Backward-compatible alias; use runtime_intervention_rate in the paper.
        "intervention_rate": float(df["runtime_intervened"].mean()),
        "modify_rate": float((df["assurance_decision"] == "modify").mean()),
        "override_rate": float((df["assurance_decision"] == "override").mean()),
        "reject_rate": float((df["assurance_decision"] == "reject").mean()),
        "mean_action_distance": float(df["action_distance"].mean()),
        "mean_battery_soc": float(df["battery_soc"].mean()),
        "mean_ev_soc": float(df["ev_soc"].mean()),
        "mean_proposal_latency_ms": float(df["proposal_latency_ms"].mean()),
        "median_proposal_latency_ms": float(df["proposal_latency_ms"].median()),
        "p95_proposal_latency_ms": float(df["proposal_latency_ms"].quantile(0.95)),
        "mean_safety_layer_latency_ms": float(df["safety_layer_latency_ms"].mean()),
        "median_safety_layer_latency_ms": float(df["safety_layer_latency_ms"].median()),
        "p95_safety_layer_latency_ms": float(df["safety_layer_latency_ms"].quantile(0.95)),
        "p99_safety_layer_latency_ms": float(df["safety_layer_latency_ms"].quantile(0.99)),
        "mean_measured_compute_latency_ms": float(df["measured_compute_latency_ms"].mean()),
        "median_measured_compute_latency_ms": float(df["measured_compute_latency_ms"].median()),
        "p95_measured_compute_latency_ms": float(df["measured_compute_latency_ms"].quantile(0.95)),
        "p99_measured_compute_latency_ms": float(df["measured_compute_latency_ms"].quantile(0.99)),
        "mean_pf_calls_total": float(df["pf_calls_total"].mean()),
        "max_pf_calls_total": float(df["pf_calls_total"].max()),
        "mean_pf_calls_safety_layer": float(df["pf_calls_safety_layer"].mean()),
        "max_pf_calls_safety_layer": float(df["pf_calls_safety_layer"].max()),
    }
