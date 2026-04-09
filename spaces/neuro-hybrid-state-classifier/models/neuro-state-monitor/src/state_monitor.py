# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""State monitor: record membrane potential traces from state signals."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim.visuals import VisualSpec

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


class StateMonitor(BioModule):
    """Record membrane potential traces from state signals.

    Receives `state` signals (from IzhikevichPopulation or HodgkinHuxleyPopulation)
    and produces a timeseries VisualSpec of voltage traces.

    Parameters:
        max_points: Maximum points per series (oldest dropped).
    """

    def __init__(self, max_points: int = 5000, min_dt: float = 0.001) -> None:
        self.min_dt = min_dt
        self.max_points = max_points
        self._series: Dict[int, List[List[float]]] = {}  # neuron_idx -> [[t, v], ...]
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return {"state"}

    def outputs(self) -> Set[str]:
        return {"voltage_summary"}

    def reset(self) -> None:
        self._series = {}
        self._outputs = {}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get("state")
        if signal is None:
            return
        payload = signal.value or {}
        t = float(payload.get("t", signal.time))
        indices = payload.get("indices", [])
        v_vals = payload.get("v", [])

        for idx, v in zip(indices, v_vals):
            if idx not in self._series:
                self._series[idx] = []
            self._series[idx].append([t, float(v)])

            # Trim if over limit
            if len(self._series[idx]) > self.max_points:
                self._series[idx] = self._series[idx][-self.max_points:]

    def advance_to(self, t: float) -> None:
        # Emit a compact summary signal (latest samples + bounded recent traces).
        latest: Dict[int, float] = {}
        recent: Dict[int, List[List[float]]] = {}
        for idx, points in self._series.items():
            if points:
                latest[idx] = float(points[-1][1])
                recent[idx] = points[-min(200, len(points)):]

        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "voltage_summary": BioSignal(
                source=source,
                name="voltage_summary",
                value={
                    "t": t,
                    "latest_v_mV": latest,
                    "recent": recent,
                },
                time=t,
                metadata=SignalMetadata(description="Voltage trace summary", kind="state", units="mV"),
            )
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def visualize(self) -> Optional["VisualSpec"]:
        if not self._series:
            return None

        series_list = []
        for idx in sorted(self._series.keys()):
            series_list.append({
                "name": f"Neuron {idx} Vm (mV)",
                "points": self._series[idx],
            })

        return {
            "render": "timeseries",
            "data": {"series": series_list},
            "description": (
                "Membrane potential (Vm) traces of sampled neurons over time. "
                "Sharp upward deflections reaching ~30mV indicate action potentials (spikes). "
                "After spiking, the membrane resets to a lower potential before gradually depolarizing again."
            ),
        }
