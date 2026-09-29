from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from safe_der.agents.policies import resource_bounds
from safe_der.experiment import load_config
from safe_der.grid.model import DistributionGrid
from safe_der.scenarios.synthetic import generate_profile
from safe_der.types import DispatchAction, ResourceState


def candidates(cfg: dict, state: ResourceState, row) -> list[DispatchAction]:
    b = resource_bounds(cfg, state, float(row["ev_connected"]))
    bvals = np.linspace(b["battery_mw"][0], b["battery_mw"][1], 5)
    evals = np.linspace(b["ev_mw"][0], b["ev_mw"][1], 5)
    fvals = np.linspace(0.0, b["flex_load_mw"][1], 3)
    # Curtailment extremes + midpoint; enough for a conservative design-screening oracle.
    pvals = [0.0, 0.5 * float(row["pv_mw"]), float(row["pv_mw"])]
    wvals = [0.0, 0.5 * float(row["wind_mw"]), float(row["wind_mw"])]
    out=[]
    for bp in bvals:
        for ep in evals:
            for fp in fvals:
                for pc in pvals:
                    for wc in wvals:
                        out.append(DispatchAction(float(bp), float(ep), float(fp), float(pc), float(wc)))
    return out


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/smoke.yaml')
    ap.add_argument('--scenario', action='append', dest='scenarios')
    ap.add_argument('--seed', type=int, default=7)
    args=ap.parse_args()
    cfg=load_config(args.config)
    grid=DistributionGrid(cfg)
    scenarios=args.scenarios or cfg['experiment']['scenarios']
    a=cfg['assets']
    for sc in scenarios:
        profile=generate_profile(cfg, sc, args.seed)
        # This is an instantaneous feasibility screen using fresh nominal SoC at every step.
        # It tests whether the grid state is controllable in principle, not whether a causal
        # controller preserved enough energy to reach that action later in the day.
        feasible=[]
        for _, row in profile.iterrows():
            state=ResourceState(float(a['battery_soc_initial']), float(a['ev_soc_initial']))
            ok=False
            for action in candidates(cfg,state,row):
                if grid.assess(float(row['demand_factor']),float(row['pv_mw']),float(row['wind_mw']),action).safe:
                    ok=True; break
            feasible.append(ok)
        rate=float(np.mean(feasible))
        bad=[i for i,x in enumerate(feasible) if not x]
        print(f"{sc}: coarse instantaneous feasible rate={rate:.3f}; infeasible_steps={bad[:20]}{'...' if len(bad)>20 else ''}")

if __name__=='__main__':
    main()
