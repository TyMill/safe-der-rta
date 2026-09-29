from safe_der.assurance.runtime import RuntimeAssurance
from safe_der.experiment import load_config
from safe_der.grid.model import DistributionGrid
from safe_der.types import DispatchAction, ResourceState


def test_assurance_returns_valid_decision():
    cfg = load_config("configs/smoke.yaml")
    grid = DistributionGrid(cfg)
    ra = RuntimeAssurance(cfg, grid)
    row = {"demand_factor": 1.0, "pv_mw": 1.0, "wind_mw": 0.5, "ev_connected": 1.0}
    state = ResourceState(0.55, 0.50)
    r = ra.evaluate(DispatchAction(battery_mw=0.5, ev_mw=0.2), row, state)
    assert r.decision in {"approve", "modify", "reject"}


def test_assured_proposal_preserves_configured_reserve():
    from safe_der.coordination.controllers import propose

    cfg = load_config("configs/smoke.yaml")
    grid = DistributionGrid(cfg)
    # Exactly at the configured battery/EV reserve: routine policy must not discharge.
    state = ResourceState(
        float(cfg["assurance"]["battery_reserve_soc"]),
        float(cfg["assurance"]["ev_reserve_soc"]),
    )
    row = {
        "demand_factor": 1.2, "pv_mw": 0.0, "wind_mw": 0.0,
        "price": 1.0, "ev_connected": 1.0,
    }
    action = propose("assured", row, cfg, state, grid)
    assert action.battery_mw <= 1e-9
    assert action.ev_mw <= 1e-9
