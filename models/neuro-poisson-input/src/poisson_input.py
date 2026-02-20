# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Poisson spike generator module for neuron simulations."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim import BioWorld

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


class PoissonInput(BioModule):
    """Generate spikes according to a Poisson process.

    Emits `spikes` topic with payload `{"t": float, "ids": [int, ...]}` on each STEP.

    Parameters:
        n: Number of independent spike sources (neuron indices 0..n-1).
        rate_hz: Firing rate in Hz for each source.
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, n: int = 100, rate_hz: float = 10.0, seed: Optional[int] = None, min_dt: float = 0.001) -> None:
        self.min_dt = min_dt
        self.n = n
        self.rate_hz = rate_hz
        self.seed = seed
        self._rng: random.Random = random.Random(seed)
        self._time: float = 0.0
        self._spike_counts: List[int] = []  # spike count per step for visualization
        self._spike_times: List[float] = []  # time points
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return set()

    def outputs(self) -> Set[str]:
        return {"spikes"}

    def reset(self) -> None:
        """Reset internal state for a new simulation run."""
        self._rng = random.Random(self.seed)
        self._time = 0.0
        self._spike_counts = []
        self._spike_times = []

    def advance_to(self, t: float) -> None:
        dt = t - self._time if t > self._time else self.min_dt
        self._time = t

        # For each neuron, probability of spike in interval dt
        # P(spike) = 1 - exp(-rate * dt) ~ rate * dt for small dt
        prob = self.rate_hz * dt
        spiked_ids: List[int] = []
        for i in range(self.n):
            if self._rng.random() < prob:
                spiked_ids.append(i)

        # Record for visualization
        self._spike_times.append(t)
        self._spike_counts.append(len(spiked_ids))

        source_name = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "spikes": BioSignal(
                source=source_name,
                name="spikes",
                value=spiked_ids,
                time=t,
                metadata=SignalMetadata(units="Hz", description="Poisson spike events", kind="event"),
            )
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def visualize(self) -> Optional[Dict[str, Any]]:
        """Return a timeseries visualization of spike counts over time."""
        if not self._spike_times:
            return None
        points = list(zip(self._spike_times, self._spike_counts))
        return {
            "render": "timeseries",
            "data": {
                "series": [{"name": "spike_count", "points": points}],
                "title": f"PoissonInput (n={self.n}, rate={self.rate_hz}Hz)",
            },
            "description": (
                f"Poisson spike generator with {self.n} independent sources firing at {self.rate_hz}Hz. "
                "Shows number of spikes generated per timestep. Poisson processes model random, "
                "independent spike arrival typical of background synaptic input."
            ),
        }
