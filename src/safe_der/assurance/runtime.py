from __future__ import annotations

import math

from safe_der.agents.policies import clip_action, resource_bounds
from safe_der.types import AssuranceResult, DispatchAction, ResourceState


def action_distance(a: DispatchAction, b: DispatchAction) -> float:
    return math.sqrt(sum((a.as_dict()[k] - b.as_dict()[k]) ** 2 for k in a.as_dict()))


def _lerp(a: DispatchAction, b: DispatchAction, alpha: float) -> DispatchAction:
    ad, bd = a.as_dict(), b.as_dict()
    return DispatchAction(**{k: ad[k] + alpha * (bd[k] - ad[k]) for k in ad})


def _violation_score(ass) -> float:
    if not ass.converged:
        return 1e9
    # Normalized voltage and thermal excess. This is used only to choose a best-effort
    # emergency fallback if the instantaneous feasible set contains no fully safe action.
    return (ass.voltage_violation / 0.01) ** 2 + (ass.thermal_violation / 10.0) ** 2


class RuntimeAssurance:
    def __init__(self, cfg: dict, grid):
        self.cfg = cfg
        self.grid = grid

    def _reserve_guard(self, proposed: DispatchAction, state: ResourceState, row) -> DispatchAction:
        """Preserve configurable dispatchable-energy reserves during normal operation.

        The reserve is a resilience/liveness guard, not the physical SoC minimum. Emergency
        overrides may consume this reserve when required to restore an electrical constraint.
        """
        ass_cfg = self.cfg.get("assurance", {})
        dt = float(self.cfg["experiment"]["step_minutes"]) / 60.0
        eta = float(self.cfg["assets"]["battery_efficiency"])
        bcap = float(self.cfg["assets"]["battery_energy_mwh"])
        ecap = float(self.cfg["assets"]["ev_energy_mwh"])
        b_res = float(ass_cfg.get("battery_reserve_soc", self.cfg["assets"]["battery_soc_min"]))
        e_res = float(ass_cfg.get("ev_reserve_soc", 0.20))

        b_max_reserve = max(0.0, (state.battery_soc - b_res) * bcap * eta / dt)
        e_max_reserve = max(0.0, (state.ev_soc - e_res) * ecap * eta / dt)
        guarded = DispatchAction(
            battery_mw=min(proposed.battery_mw, b_max_reserve) if proposed.battery_mw > 0 else proposed.battery_mw,
            ev_mw=min(proposed.ev_mw, e_max_reserve) if proposed.ev_mw > 0 else proposed.ev_mw,
            flex_load_mw=proposed.flex_load_mw,
            pv_curtail_mw=proposed.pv_curtail_mw,
            wind_curtail_mw=proposed.wind_curtail_mw,
        )
        return clip_action(guarded, self.cfg, state, row)

    def _emergency_candidates(self, row, state: ResourceState, guarded: DispatchAction) -> list[DispatchAction]:
        """Build a small deterministic library of physically distinct emergency actions.

        Thermal overload can arise from import-side stress, reverse DER export, or a
        local branch effect. A single global heuristic is therefore insufficient.
        Candidate actions cover import relief, renewable curtailment, storage
        absorption, and mixed export relief. The best safe direction is selected by
        AC power-flow assessment, not by a guessed sign alone.
        """
        bounds = resource_bounds(self.cfg, state, float(row["ev_connected"]))
        b_lo, b_hi = bounds["battery_mw"]
        e_lo, e_hi = bounds["ev_mw"]
        f_hi = bounds["flex_load_mw"][1]
        pv = float(row["pv_mw"])
        wind = float(row["wind_mw"])

        out: list[DispatchAction] = [guarded, DispatchAction()]
        for frac in (0.25, 0.50, 0.75, 1.0):
            # Import-side relief: local generation/discharge + flexible-load reduction.
            out.append(clip_action(DispatchAction(
                battery_mw=frac * b_hi,
                ev_mw=frac * e_hi,
                flex_load_mw=frac * f_hi,
            ), self.cfg, state, row))

            # Export-side relief A: curtailment only. This avoids creating new local
            # charging flows on branches that may themselves be thermally limiting.
            out.append(clip_action(DispatchAction(
                pv_curtail_mw=frac * pv,
                wind_curtail_mw=frac * wind,
            ), self.cfg, state, row))

            # Export-side relief B: storage absorption only.
            out.append(clip_action(DispatchAction(
                battery_mw=frac * b_lo,
                ev_mw=frac * e_lo,
            ), self.cfg, state, row))

            # Export-side relief C: mixed absorption + curtailment.
            out.append(clip_action(DispatchAction(
                battery_mw=frac * b_lo,
                ev_mw=frac * e_lo,
                pv_curtail_mw=frac * pv,
                wind_curtail_mw=frac * wind,
            ), self.cfg, state, row))
        return out

    def evaluate(self, proposed: DispatchAction, row, state: ResourceState) -> AssuranceResult:
        physical = clip_action(proposed, self.cfg, state, row)
        guarded = self._reserve_guard(physical, state, row)
        guard_changed = action_distance(physical, guarded) > 1e-12

        ass = self.grid.assess(float(row["demand_factor"]), float(row["pv_mw"]), float(row["wind_mw"]), guarded)
        if ass.safe:
            decision = "modify" if guard_changed else "approve"
            return AssuranceResult(decision, physical, guarded, ass, action_distance(physical, guarded), 0)

        # Explore multiple physically plausible recovery directions. This is only
        # invoked after the guarded autonomous proposal violates a grid constraint.
        assessed = []
        for candidate in self._emergency_candidates(row, state, guarded):
            ca = self.grid.assess(float(row["demand_factor"]), float(row["pv_mw"]), float(row["wind_mw"]), candidate)
            assessed.append((candidate, ca))

        max_iter = int(self.cfg["assurance"]["repair_max_iter"])
        safe_targets = [(a, ca) for a, ca in assessed if ca.safe]
        if safe_targets:
            # Pick the nearest known safe target, then binary-search the smallest
            # interpolation from the guarded proposal that restores all constraints.
            target, target_ass = min(safe_targets, key=lambda item: action_distance(guarded, item[0]))
            lo, hi = 0.0, 1.0
            best, best_ass = target, target_ass
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                candidate = clip_action(_lerp(guarded, target, mid), self.cfg, state, row)
                cand_ass = self.grid.assess(float(row["demand_factor"]), float(row["pv_mw"]), float(row["wind_mw"]), candidate)
                if cand_ass.safe:
                    best, best_ass, hi = candidate, cand_ass, mid
                else:
                    lo = mid
            return AssuranceResult("modify", physical, best, best_ass, action_distance(physical, best), max_iter)

        # No instantaneous feasible action was found in the bounded recovery library.
        # Execute the candidate with the smallest normalized violation and explicitly
        # report a reject decision rather than silently claiming safety.
        fallback, f_ass = min(assessed, key=lambda item: _violation_score(item[1]))
        return AssuranceResult("reject", physical, fallback, f_ass, action_distance(physical, fallback), max_iter)
