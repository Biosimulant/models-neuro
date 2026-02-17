# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Step current injection module for neuron simulations."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bsim import BioWorld

from bsim import BioModule
from bsim.signals import BioSignal, SignalMetadata


class StepCurrent(BioModule):
    """Inject a constant or scheduled current.

    Emits `current` topic with payload `{"t": float, "I": float}` on each STEP.

    Parameters:
        I: Constant current amplitude (in nA or arbitrary units).
        schedule: Optional list of (start_t, end_t, I_value) tuples for time-varying current.
                  If provided, `I` is used as the default when outside scheduled intervals.
    """

    def __init__(
        self,
        I: float = 10.0,
        schedule: Optional[List[tuple]] = None,
        min_dt: float = 0.001,
    ) -> None:
        self.min_dt = min_dt
        self.I_default = I
        self.schedule = schedule or []
        self._time: float = 0.0
        self._current_history: List[List[float]] = []  # [[t, I], ...]
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return set()

    def outputs(self) -> Set[str]:
        return {"current"}

    def reset(self) -> None:
        self._time = 0.0
        self._current_history = []

    def _get_current(self, t: float) -> float:
        """Get current value at time t based on schedule."""
        for start, end, I_val in self.schedule:
            if start <= t < end:
                return float(I_val)
        return self.I_default

    def advance_to(self, t: float) -> None:
        self._time = t

        I = self._get_current(t)
        self._current_history.append([t, I])
        source_name = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "current": BioSignal(
                source=source_name,
                name="current",
                value=float(I),
                time=t,
                metadata=SignalMetadata(units="nA", description="Injected current", kind="state"),
            )
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def visualize(self) -> Optional[Dict[str, Any]]:
        """Return a timeseries visualization of injected current over time."""
        if not self._current_history:
            return None
        return {
            "render": "timeseries",
            "data": {
                "series": [{"name": "I", "points": self._current_history}],
                "title": f"StepCurrent (I={self.I_default})",
            },
            "description": (
                f"Current injection with default amplitude {self.I_default} (arbitrary units). "
                "This current is directly added to the target neuron's membrane equation, "
                "driving it toward threshold and producing spikes when sufficiently strong."
            ),
        }


# Alias for backward compatibility / alternative naming
DCInput = StepCurrent
