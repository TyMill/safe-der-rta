from __future__ import annotations

from safe_der.coordination.hard_constrained import solve_hard_constrained
from safe_der.types import GridAssessment, ResourceState


class FakeGrid:
    def __init__(self):
        self.assess_calls = 0

    def assess(self, demand_factor, pv_mw, wind_mw, action):
        self.assess_calls += 1
        # Feasible when battery support reaches 0.2 MW. Other constraints are benign.
        vv = max(0.0, 0.2 - action.battery_mw) * 0.01
        return GridAssessment(True, 0.95 - vv, 1.0, 50.0, 1.0, 0.01, vv, 0.0)


def test_hard_constrained_finds_feasible_action():
    cfg = {
        "experiment": {"step_minutes": 15},
        "network": {"voltage_min_pu": 0.95, "voltage_max_pu": 1.05, "line_loading_max_pct": 100.0},
        "assets": {
            "battery_power_mw": 0.8, "battery_energy_mwh": 2.4, "battery_soc_min": 0.15,
            "battery_soc_max": 0.9, "battery_efficiency": 0.95, "ev_power_mw": 0.55,
            "ev_energy_mwh": 1.2, "flexible_load_mw": 0.45,
        },
        "objective": {
            "import_cost_weight": 1.0, "curtailment_weight": 2.0,
            "battery_degradation_weight": 0.08, "action_smoothness_weight": 0.03,
        },
        "hard_constrained": {"maxiter": 30, "ftol": 1e-8},
    }
    row = {"demand_factor": 1.0, "pv_mw": 1.0, "wind_mw": 0.5, "price": 0.5, "ev_connected": 1.0}
    grid = FakeGrid()
    result = solve_hard_constrained(row, cfg, ResourceState(0.55, 0.5), grid)
    assert result.assessment.safe
    assert result.solver_status in {"success_feasible", "recovered_feasible"}
    assert result.evaluated_points > 0
    assert grid.assess_calls == result.evaluated_points
