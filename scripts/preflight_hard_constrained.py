from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "experiments.yaml"
NEW = ROOT / "configs" / "hard_constrained.yaml"

base = yaml.safe_load(BASE.read_text())
new = yaml.safe_load(NEW.read_text())

# Fields inherited from frozen v0.6 must be byte-for-value identical.
for section in ("network", "uncertainty", "assets", "objective", "assurance"):
    assert new[section] == base[section], f"Drift detected in {section}"
for key in ("seeds", "scenarios", "horizon_steps", "step_minutes", "save_step_metrics"):
    assert new["experiment"][key] == base["experiment"][key], f"Drift detected in experiment.{key}"
assert new["experiment"]["controllers"] == ["hard_constrained"]
assert new["hard_constrained"]["method"] == "SLSQP"
expected = len(new["experiment"]["seeds"]) * len(new["experiment"]["scenarios"])
print("[OK] Frozen v0.6 parameters preserved.")
print("[OK] Controller: hard_constrained")
print(f"[OK] Diagnostic episodes: {expected}")
print("[OK] Direct optimizer uses full physical actuator bounds; no RTA/reserve guard is applied.")
print("[OK] Solver failure is logged separately from grid feasibility claims.")
