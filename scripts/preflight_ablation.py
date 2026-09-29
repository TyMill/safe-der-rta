#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalized(cfg: dict) -> dict:
    out = deepcopy(cfg)
    out["experiment"].pop("name", None)
    out["experiment"].pop("controllers", None)
    return out


def main() -> int:
    frozen = load_config(ROOT / "configs" / "experiments.yaml")
    ablation = load_config(ROOT / "configs" / "ablation_reserve_rta.yaml")

    expected = ["reserve_only", "assured"]
    got = list(ablation["experiment"]["controllers"])
    if got != expected:
        print(f"[FAIL] Ablation controllers: expected {expected}, got {got}")
        return 1

    if normalized(frozen) != normalized(ablation):
        print("[FAIL] Ablation config drift detected outside experiment.name/controllers.")
        # Print top-level sections that differ for quick diagnosis.
        for key in frozen:
            a = normalized(frozen).get(key)
            b = normalized(ablation).get(key)
            if a != b:
                print(f"  differs: {key}")
        return 1

    n = len(ablation["experiment"]["seeds"]) * len(ablation["experiment"]["scenarios"]) * len(got)
    print("[OK] Frozen v0.6 parameters preserved.")
    print("[OK] Controllers: reserve_only, assured")
    print(f"[OK] Diagnostic episodes: {n}")
    print("[OK] No scenario/reserve/network/objective/uncertainty drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
