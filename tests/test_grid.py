from safe_der.experiment import load_config
from safe_der.grid.model import DistributionGrid
from safe_der.types import DispatchAction


def test_ieee33_powerflow_runs():
    cfg = load_config("configs/smoke.yaml")
    grid = DistributionGrid(cfg)
    a = grid.assess(0.8, 1.0, 0.5, DispatchAction())
    assert a.converged
    assert 0.0 < a.min_voltage_pu < 2.0
    assert a.max_line_loading_pct >= 0.0
