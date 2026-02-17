# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Rate monitor: compute population firing rate from spike events."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bsim.visuals import VisualSpec

from bsim import BioModule
from bsim.signals import BioSignal, SignalMetadata


class RateMonitor(BioModule):
    """Compute population firing rate from spike events.

    Receives `spikes` signals, computes instantaneous firing rate, and produces
    a timeseries VisualSpec.

    Parameters:
        window_size: Time window in seconds for rate computation.
        n_neurons: Total number of neurons (for rate normalization).
    """

    def __init__(
        self,
        window_size: float = 0.05,
        n_neurons: int = 100,
        min_dt: float = 0.001,
    ) -> None:
        self.min_dt = min_dt
        self.window_size = window_size
        self.n_neurons = n_neurons

        self._spike_times: List[float] = []
        self._rate_series: List[List[float]] = []  # [[t, rate], ...]
        self._last_t: float = 0.0
        self._last_rate: float = 0.0
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return {"spikes"}

    def outputs(self) -> Set[str]:
        return {"rate_state"}

    def reset(self) -> None:
        self._spike_times = []
        self._rate_series = []
        self._last_t = 0.0
        self._last_rate = 0.0
        self._outputs = {}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get("spikes")
        if signal is None:
            return
        t = float(signal.time)
        ids = signal.value or []

        # Record all spike times
        for _ in ids:
            self._spike_times.append(t)

        # Compute rate at this time point
        # Count spikes in [t - window, t]
        window_start = t - self.window_size
        n_in_window = sum(1 for st in self._spike_times if st >= window_start)

        # Rate = (spikes in window) / (window_size * n_neurons) in Hz
        rate = n_in_window / (self.window_size * self.n_neurons) if self.n_neurons > 0 else 0.0
        self._rate_series.append([t, rate])
        self._last_t = t
        self._last_rate = float(rate)

        # Trim old spike times for memory efficiency
        self._spike_times = [st for st in self._spike_times if st >= window_start]

    def advance_to(self, t: float) -> None:
        source = getattr(self, "_world_name", self.__class__.__name__)
        recent = self._rate_series[-min(500, len(self._rate_series)):] if self._rate_series else []
        self._outputs = {
            "rate_state": BioSignal(
                source=source,
                name="rate_state",
                value={
                    "t": t,
                    "rate_hz": self._last_rate,
                    "window_s": float(self.window_size),
                    "n_neurons": int(self.n_neurons),
                    "recent": [[float(tt), float(r)] for tt, r in recent],
                },
                time=t,
                metadata=SignalMetadata(description="Population firing rate state", kind="state", units="Hz"),
            )
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def visualize(self) -> Optional["VisualSpec"]:
        if not self._rate_series:
            return None

        return {
            "render": "timeseries",
            "data": {
                "series": [
                    {"name": "Population Rate (Hz)", "points": self._rate_series}
                ]
            },
            "description": (
                f"Population firing rate computed over a {self.window_size*1000:.0f}ms sliding window. "
                f"Rate = (spikes in window) / (window size \u00d7 {self.n_neurons} neurons). "
                "Higher values indicate more synchronized or active network states."
            ),
        }
