from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trajectory_analysis", ROOT / "scripts" / "analyze_trajectory_feasibility.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_pre_state_reconstruction_from_post_state():
    cfg = {"assets": {"battery_soc_initial": 0.55, "ev_soc_initial": 0.50}}
    df = pd.DataFrame({"step": [0, 1, 2], "battery_soc": [0.50, 0.45, 0.40], "ev_soc": [0.48, 0.46, 0.44]})
    out = MOD._ensure_pre_state(df, cfg)
    assert out["battery_soc_before"].tolist() == [0.55, 0.50, 0.45]
    assert out["ev_soc_before"].tolist() == [0.50, 0.48, 0.46]
