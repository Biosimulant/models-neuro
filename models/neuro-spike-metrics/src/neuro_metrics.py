# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Neuro metrics: compute summary statistics from spike data."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim.visuals import VisualSpec

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


class NeuroMetrics(BioModule):
    """Compute summary statistics from spike data.

    Receives `spikes` signals and produces a table VisualSpec with metrics:
    - Total spike count
    - Mean firing rate
    - Number of active neurons
    - Coefficient of variation (CV) of ISIs

    Parameters:
        n_neurons: Total number of neurons (for rate calculation).
    """

    def __init__(self, n_neurons: int = 100, min_dt: float = 0.001) -> None:
        self.min_dt = min_dt
        self.n_neurons = n_neurons

        self._spike_count: int = 0
        self._active_neurons: Set[int] = set()
        self._spike_times_by_neuron: Dict[int, List[float]] = {}
        self._t_start: Optional[float] = None
        self._t_end: float = 0.0
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return {"spikes"}

    def outputs(self) -> Set[str]:
        return {"metrics"}

    def reset(self) -> None:
        self._spike_count = 0
        self._active_neurons = set()
        self._spike_times_by_neuron = {}
        self._t_start = None
        self._t_end = 0.0
        self._outputs = {}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get("spikes")
        if signal is None:
            return
        t = float(signal.time)
        ids = signal.value or []

        if self._t_start is None:
            self._t_start = t
        self._t_end = t

        for nid in ids:
            nid = int(nid)
            self._spike_count += 1
            self._active_neurons.add(nid)

            if nid not in self._spike_times_by_neuron:
                self._spike_times_by_neuron[nid] = []
            self._spike_times_by_neuron[nid].append(t)

    def advance_to(self, t: float) -> None:
        # Emit incremental metrics as a state signal to enable downstream analysis.
        self._t_end = max(self._t_end, float(t))
        duration = self._t_end - (self._t_start or 0.0) if self._t_start is not None else 0.0
        mean_rate_hz = (
            self._spike_count / (duration * self.n_neurons)
            if duration > 0 and self.n_neurons > 0
            else 0.0
        )
        cv = self._compute_cv()

        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "metrics": BioSignal(
                source=source,
                name="metrics",
                value={
                    "t": float(t),
                    "total_spikes": int(self._spike_count),
                    "active_neurons": int(len(self._active_neurons)),
                    "duration_s": float(duration),
                    "mean_rate_hz": float(mean_rate_hz),
                    "isi_cv": None if cv is None else float(cv),
                    "n_neurons": int(self.n_neurons),
                },
                time=float(t),
                metadata=SignalMetadata(description="Neural spike stream summary metrics", kind="state"),
            )
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def _compute_cv(self) -> Optional[float]:
        """Compute coefficient of variation of inter-spike intervals."""
        all_isis: List[float] = []
        for times in self._spike_times_by_neuron.values():
            if len(times) >= 2:
                sorted_times = sorted(times)
                for i in range(1, len(sorted_times)):
                    isi = sorted_times[i] - sorted_times[i - 1]
                    if isi > 0:
                        all_isis.append(isi)

        if len(all_isis) < 2:
            return None

        mean_isi = sum(all_isis) / len(all_isis)
        if mean_isi <= 0:
            return None

        variance = sum((isi - mean_isi) ** 2 for isi in all_isis) / len(all_isis)
        std_isi = variance ** 0.5
        return std_isi / mean_isi

    def visualize(self) -> Optional["VisualSpec"]:
        duration = self._t_end - (self._t_start or 0.0) if self._t_start is not None else 0.0

        # Mean rate in Hz (spikes per second per neuron)
        if duration > 0 and self.n_neurons > 0:
            mean_rate = self._spike_count / (duration * self.n_neurons)
        else:
            mean_rate = 0.0

        cv = self._compute_cv()

        rows = [
            ["Total Spikes", str(self._spike_count)],
            ["Active Neurons", str(len(self._active_neurons))],
            ["Duration (s)", f"{duration:.3f}"],
            ["Mean Rate (Hz)", f"{mean_rate:.2f}"],
            ["ISI CV", f"{cv:.3f}" if cv is not None else "N/A"],
        ]

        return {
            "render": "table",
            "data": {
                "columns": ["Metric", "Value"],
                "rows": rows,
            },
            "description": (
                "Summary statistics of neural activity. "
                "ISI CV (coefficient of variation of inter-spike intervals) measures firing regularity: "
                "CV\u22480 = perfectly regular, CV\u22481 = Poisson-like irregular, CV>1 = bursty firing patterns."
            ),
        }
