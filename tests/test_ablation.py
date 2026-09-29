from __future__ import annotations

from safe_der.coordination.controllers import propose
from safe_der.grid.model import DistributionGrid
from safe_der.scenarios.synthetic import deterministic_normal_profile
from safe_der.types import ResourceState
from safe_der.experiment import load_config, run_episode


def test_reserve_only_and_assured_share_proposal_policy():
    cfg = load_config("configs/ablation_smoke.yaml")
    grid = DistributionGrid(cfg)
    row = deterministic_normal_profile(cfg).iloc[40]
    state = ResourceState(cfg["assets"]["battery_soc_initial"], cfg["assets"]["ev_soc_initial"])
    a = propose("reserve_only", row, cfg, state, grid)
    b = propose("assured", row, cfg, state, grid)
    assert a.as_dict() == b.as_dict()


def test_instrumentation_and_pf_counts_present():
    cfg = load_config("configs/ablation_smoke.yaml")
    # Shorten only inside the test; this does not modify the experiment config on disk.
    cfg["experiment"]["horizon_steps"] = 2
    df, _ = run_episode(cfg, "assured", "normal", 7)
    required = {
        "proposal_latency_ms", "safety_layer_latency_ms", "grid_evaluation_latency_ms",
        "decision_latency_ms", "measured_compute_latency_ms", "pf_calls_proposal",
        "pf_calls_safety_layer", "pf_calls_evaluation", "pf_calls_total",
    }
    assert required.issubset(df.columns)
    assert (df["pf_calls_safety_layer"] >= 1).all()
    assert (df["pf_calls_total"] == df["pf_calls_proposal"] + df["pf_calls_safety_layer"] + df["pf_calls_evaluation"]).all()
