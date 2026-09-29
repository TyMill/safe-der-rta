
[![DOI](https://zenodo.org/badge/1394444183.svg)](https://doi.org/10.5281/zenodo.23034031)

# Safe-DER-RTA

Reproducible research code and archived outputs for **Assurance-Governed Autonomous Flexibility Dispatch in Renewable-Rich Distribution Networks under Uncertainty**.

> Historical note: the Python distribution/package directory retains the name `safe-der-mas` for continuity with the frozen experimental campaigns. The manuscript and release use **autonomous flexibility dispatch** because the implemented controller proposes a coordinated joint DER action rather than independent learned agent policies.

## What this repository evaluates

A performance-oriented controller proposes a joint action for BESS, EV/V2G, flexible demand, PV curtailment, and wind curtailment. An AC-power-flow runtime-assurance (RTA) layer then returns **Approve**, **Modify**, or **Reject + best-effort fallback**. Reject means that the bounded online search did not certify a safe action; it is not itself a safety guarantee.

Grid safety requires converged AC power flow, bus voltages in 0.95–1.05 p.u., and line loading at or below 100%, together with implemented resource bounds.

## Frozen campaigns

### v0.6.0 — main campaign
- IEEE 33-bus benchmark
- 20 seeds × 7 scenarios × 5 controllers = **700 episodes**
- 96 quarter-hour steps per episode = **67,200 steps**
- controllers: `passive`, `rule_based`, `unconstrained`, `penalty`, `assured`
- scenarios: `normal`, `solar_surge`, `renewable_drop`, `demand_spike`, `ev_coincidence`, `compound`, `extreme_compound`

### v0.6.1 — reserve/RTA ablation
- 20 seeds × 7 scenarios × 2 controllers = **280 episodes / 26,880 steps**
- `reserve_only` uses the same reserve-aware proposal policy as `assured` but executes without RTA
- instrumented `assured` rerun reproduced the frozen v0.6.0 episode metrics exactly
- logs proposal/RTA/evaluation latency and AC power-flow call counts

### v0.6.2 — hard-constrained diagnostic comparator
- 20 seeds × 7 scenarios × `hard_constrained` = **140 episodes / 13,440 steps**
- one-step SLSQP with explicit AC voltage and thermal constraints
- feasible points found during optimization are retained
- `best_effort_no_feasible_found` means the solver did not identify a feasible action; it is **not** proof of continuous-space infeasibility

## Post-hoc trajectory feasibility

Residual unsafe states are checked with a separately defined coarse state-aware lattice:

- 5 BESS levels × 5 EV levels × 3 flexible-load levels × 3 PV-curtailment levels × 3 wind-curtailment levels = **675 candidate actions maximum per state**.
- `safe_executed`: executed action is safe.
- `recoverable_but_missed`: executed action is unsafe but a safe lattice witness exists.
- `no_safe_lattice_witness`: no safe action is found in the tested lattice.

`no_safe_lattice_witness` is deliberately **not** called physical infeasibility. The lattice is finite and uses the same network model; it cannot prove that the continuous feasible set is empty.

## Main reported findings

- `assured` achieved 100% grid-safe operation in five scenarios and 95.42% in `demand_spike`.
- The reserve-only ablation isolates an incremental RTA safety gain of 19.90–40.73 percentage points in the five fully recoverable scenarios and 33.65 points in `demand_spike`.
- The hard-constrained comparator achieved 82.24–84.90% safety in the six non-extreme scenarios and 64.11% in `extreme_compound`.
- Of 2,535 unsafe hard-constrained steps, 2,534 (99.96%) had no safe witness in the same 675-action lattice; one case was `recoverable_but_missed`.
- Timing results are empirical and hardware-specific; they are not worst-case execution-time guarantees.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -q
```

Main campaign:

```bash
python scripts/run_experiments.py --config configs/experiments.yaml
python scripts/summarize_results.py --run-dir results/latest
```

Reserve/RTA ablation:

```bash
python scripts/run_experiments.py --config configs/ablation_reserve_rta.yaml
python scripts/summarize_ablation.py --run-dir results/latest
```

Hard-constrained comparator:

```bash
python scripts/run_experiments.py --config configs/hard_constrained.yaml
python scripts/summarize_hard_constrained.py --run-dir results/latest
```

Hard-constrained trajectory diagnostic:

```bash
python scripts/analyze_hard_constrained_feasibility.py \
  --run-dir results/latest \
  --controller hard_constrained \
  --workers 8
```

## Archived outputs

The release bundle contains the exact CSV/YAML/JSON outputs used for the manuscript under `results_archive/`. Historical raw outputs are retained; corrected/normalized post-hoc summaries use `no_safe_lattice_witness` terminology and `N/A` when there are zero unsafe steps.

## Reproducibility and environment

Random seeds and resolved configurations are archived with every campaign. Historical manifests record Python 3.13.9 on macOS arm64. Before the permanent DOI deposit, capture the exact package environment on the original experiment machine:

```bash
python3 scripts/capture_environment.py
```

Wall-clock timing values depend on hardware, OS, solver/library versions, and system load and should not be expected to reproduce bit-for-bit on another platform.

## Scope

This is a synthetic IEEE 33-bus research benchmark, not a production grid-control system. It uses active-power flexibility, a one-step AC-power-flow assurance check, a finite online repair library, and a finite post-hoc feasibility lattice. The hard-constrained comparator is a myopic one-step SLSQP baseline, not multi-period AC-OPF/MPC.

## Citation

See `CITATION.cff`. The permanent DOI will be added after the Zenodo deposit is created.

## License

MIT; see `LICENSE`.
