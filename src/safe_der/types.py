from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Decision = Literal["approve", "modify", "override", "reject"]


@dataclass
class DispatchAction:
    battery_mw: float = 0.0  # + discharge, - charge
    ev_mw: float = 0.0       # + V2G discharge, - charging
    flex_load_mw: float = 0.0  # + load reduction, - load increase
    pv_curtail_mw: float = 0.0
    wind_curtail_mw: float = 0.0

    def scaled(self, factor: float) -> "DispatchAction":
        return DispatchAction(**{k: v * factor for k, v in self.as_dict().items()})

    def as_dict(self) -> dict[str, float]:
        return {
            "battery_mw": self.battery_mw,
            "ev_mw": self.ev_mw,
            "flex_load_mw": self.flex_load_mw,
            "pv_curtail_mw": self.pv_curtail_mw,
            "wind_curtail_mw": self.wind_curtail_mw,
        }


@dataclass
class ResourceState:
    battery_soc: float
    ev_soc: float
    previous_action: DispatchAction = field(default_factory=DispatchAction)


@dataclass
class ExogenousState:
    demand_factor: float
    pv_mw: float
    wind_mw: float
    price: float
    ev_connected: float


@dataclass
class GridAssessment:
    converged: bool
    min_voltage_pu: float
    max_voltage_pu: float
    max_line_loading_pct: float
    grid_import_mw: float
    losses_mw: float
    voltage_violation: float
    thermal_violation: float

    @property
    def safe(self) -> bool:
        return self.converged and self.voltage_violation <= 1e-9 and self.thermal_violation <= 1e-9


@dataclass
class AssuranceResult:
    decision: Decision
    proposed: DispatchAction
    executed: DispatchAction
    assessment: GridAssessment
    action_distance: float
    repair_iterations: int = 0
