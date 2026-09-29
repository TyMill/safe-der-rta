from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
# When copied to repo/scripts, repo root is parent of scripts.
if ROOT.name == 'scripts':
    REPO = ROOT.parent
else:
    REPO = Path.cwd()
SRC = REPO / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import yaml

from safe_der.agents.policies import resource_bounds
from safe_der.grid.model import DistributionGrid
from safe_der.scenarios.synthetic import generate_profile
from safe_der.types import DispatchAction, ResourceState


def _load_cfg(run_dir: Path) -> dict:
    with open(run_dir / 'config_resolved.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _ensure_pre_state(group: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = group.sort_values('step').copy()
    if 'battery_soc_before' in g.columns and 'ev_soc_before' in g.columns:
        return g
    a = cfg['assets']
    b_before = [float(a['battery_soc_initial'])]
    e_before = [float(a['ev_soc_initial'])]
    if len(g) > 1:
        b_before.extend(g['battery_soc'].iloc[:-1].astype(float).tolist())
        e_before.extend(g['ev_soc'].iloc[:-1].astype(float).tolist())
    g['battery_soc_before'] = b_before
    g['ev_soc_before'] = e_before
    return g


def _candidate_actions(cfg: dict, state: ResourceState, row: pd.Series):
    # EXACT same 5 x 5 x 3 x 3 x 3 = 675 state-aware lattice as v0.6 analysis.
    b = resource_bounds(cfg, state, float(row['ev_connected']))
    bvals = np.linspace(b['battery_mw'][0], b['battery_mw'][1], 5)
    evals = np.linspace(b['ev_mw'][0], b['ev_mw'][1], 5)
    fvals = np.linspace(0.0, b['flex_load_mw'][1], 3)
    pvals = (0.0, 0.5 * float(row['pv_mw']), float(row['pv_mw']))
    wvals = (0.0, 0.5 * float(row['wind_mw']), float(row['wind_mw']))
    for bp in bvals:
        for ep in evals:
            for fp in fvals:
                for pc in pvals:
                    for wc in wvals:
                        yield DispatchAction(float(bp), float(ep), float(fp), float(pc), float(wc))


def _analyze_episode(payload):
    cfg, controller, scenario, seed, records = payload
    group = pd.DataFrame.from_records(records)
    group = _ensure_pre_state(group, cfg)
    if 'ev_connected' not in group.columns:
        prof = generate_profile(cfg, str(scenario), int(seed)).set_index('step')
        group['ev_connected'] = group['step'].map(prof['ev_connected'])

    grid = DistributionGrid(cfg)
    detailed = []
    for _, row in group.iterrows():
        if bool(row['grid_safe']):
            found = True
            witness = DispatchAction(
                float(row['battery_mw']), float(row['ev_mw']),
                float(row['flex_load_mw']), float(row['pv_curtail_mw']),
                float(row['wind_curtail_mw']),
            )
            n_tested = 0
            status = 'safe_executed'
        else:
            state = ResourceState(float(row['battery_soc_before']), float(row['ev_soc_before']))
            found = False
            witness = None
            n_tested = 0
            for action in _candidate_actions(cfg, state, row):
                n_tested += 1
                ass = grid.assess(
                    float(row['demand_factor']), float(row['pv_mw']),
                    float(row['wind_mw']), action,
                )
                if ass.safe:
                    found = True
                    witness = action
                    break
            status = 'recoverable_but_missed' if found else 'no_safe_lattice_witness'

        wd = witness.as_dict() if witness is not None else {k: np.nan for k in DispatchAction().as_dict()}
        detailed.append({
            'controller': controller,
            'scenario': scenario,
            'seed': int(seed),
            'step': int(row['step']),
            'hour': float(row['hour']),
            'grid_safe': bool(row['grid_safe']),
            'trajectory_state_feasible': bool(found),
            'trajectory_status': status,
            'battery_soc_before': float(row['battery_soc_before']),
            'ev_soc_before': float(row['ev_soc_before']),
            'candidate_actions_tested': int(n_tested),
            **{f'witness_{k}': v for k, v in wd.items()},
        })
    return detailed


def analyze(run_dir: Path, controller: str, workers: int):
    cfg = _load_cfg(run_dir)
    steps = pd.read_csv(run_dir / 'step_metrics.csv')
    steps = steps.loc[steps['controller'] == controller].copy()
    if steps.empty:
        raise ValueError(f'No rows for controller={controller!r}')

    payloads = []
    for (scenario, seed), g in steps.groupby(['scenario', 'seed'], sort=True):
        payloads.append((cfg, controller, str(scenario), int(seed), g.to_dict('records')))

    all_rows = []
    if workers <= 1:
        for i, p in enumerate(payloads, 1):
            all_rows.extend(_analyze_episode(p))
            print(f'[{i}/{len(payloads)}] {p[2]} seed={p[3]}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_analyze_episode, p): (p[2], p[3]) for p in payloads}
            done = 0
            for fut in as_completed(futs):
                scenario, seed = futs[fut]
                all_rows.extend(fut.result())
                done += 1
                print(f'[{done}/{len(payloads)}] {scenario} seed={seed}', flush=True)

    detail = pd.DataFrame(all_rows).sort_values(['scenario', 'seed', 'step']).reset_index(drop=True)
    summary_rows = []
    for (scenario, seed), g in detail.groupby(['scenario', 'seed'], sort=True):
        unsafe = ~g['grid_safe']
        n_unsafe = int(unsafe.sum())
        no_witness = int((g['trajectory_status'] == 'no_safe_lattice_witness').sum())
        missed = int((g['trajectory_status'] == 'recoverable_but_missed').sum())
        summary_rows.append({
            'controller': controller,
            'scenario': scenario,
            'seed': int(seed),
            'steps': int(len(g)),
            'grid_safe_rate': float(g['grid_safe'].mean()),
            'trajectory_state_feasible_rate': float(g['trajectory_state_feasible'].mean()),
            'unsafe_steps': n_unsafe,
            'no_safe_lattice_witness_steps': no_witness,
            'recoverable_but_missed_steps': missed,
            'no_witness_share_of_unsafe_steps': (float(no_witness / n_unsafe) if n_unsafe else np.nan),
        })
    return detail, pd.DataFrame(summary_rows)


def main():
    p = argparse.ArgumentParser(description='Parallel state-aware feasibility diagnostic for hard-constrained baseline.')
    p.add_argument('--run-dir', default='results/latest')
    p.add_argument('--controller', default='hard_constrained')
    p.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    detail, summary = analyze(run_dir, args.controller, args.workers)

    detail_path = run_dir / 'trajectory_feasibility.csv'
    summary_path = run_dir / 'trajectory_feasibility_summary.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    by = summary.groupby('scenario').agg(
        seeds=('seed', 'count'),
        mean_grid_safe_rate=('grid_safe_rate', 'mean'),
        mean_trajectory_state_feasible_rate=('trajectory_state_feasible_rate', 'mean'),
        total_unsafe_steps=('unsafe_steps', 'sum'),
        total_no_safe_lattice_witness_steps=('no_safe_lattice_witness_steps', 'sum'),
        total_recoverable_but_missed_steps=('recoverable_but_missed_steps', 'sum'),
    )
    by['no_witness_share_of_unsafe'] = (
        by['total_no_safe_lattice_witness_steps'] / by['total_unsafe_steps'].replace(0, np.nan)
    )
    print('\n' + by.to_string(float_format=lambda x: f'{x:.4f}'))
    print(f'\nSaved: {detail_path}')
    print(f'Saved: {summary_path}')


if __name__ == '__main__':
    main()
