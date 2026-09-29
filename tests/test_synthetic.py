from safe_der.experiment import load_config
from safe_der.scenarios.synthetic import generate_profile


def test_profile_is_reproducible():
    cfg = load_config("configs/smoke.yaml")
    a = generate_profile(cfg, "compound", 7)
    b = generate_profile(cfg, "compound", 7)
    assert a.equals(b)
    assert len(a) == cfg["experiment"]["horizon_steps"]
    assert (a[["pv_mw", "wind_mw", "demand_factor"]] >= 0).all().all()
