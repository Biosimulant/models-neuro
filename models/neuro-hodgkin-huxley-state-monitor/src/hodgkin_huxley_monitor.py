# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""HH State Monitor: detailed visualization of Hodgkin-Huxley neuron state.

Produces 3 visualization panels:
1. Membrane potential V(t)
2. Gating variables m(t), h(t), n(t)
3. Ionic currents I_Na(t), I_K(t), I_L(t)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bsim.visuals import VisualSpec

from bsim import BioModule
from bsim.signals import BioSignal, SignalMetadata


class HHStateMonitor(BioModule):
    """Detailed state monitor for Hodgkin-Huxley neurons.

    Receives `state` signals from HodgkinHuxleyPopulation and produces
    3-panel visualization: V(t), gating variables, and ionic currents.

    Parameters:
        max_points: Maximum data points per series (oldest dropped).
        neuron_index: Which neuron to monitor (from the sampled indices).
    """

    def __init__(
        self,
        max_points: int = 5000,
        neuron_index: int = 0,
        min_dt: float = 0.0001,
    ) -> None:
        self.min_dt = min_dt
        self.max_points = max_points
        self.neuron_index = neuron_index

        # Time series data
        self._v_series: List[List[float]] = []      # [[t, V], ...]
        self._m_series: List[List[float]] = []      # [[t, m], ...]
        self._h_series: List[List[float]] = []      # [[t, h], ...]
        self._n_series: List[List[float]] = []      # [[t, n], ...]
        self._I_Na_series: List[List[float]] = []   # [[t, I_Na], ...]
        self._I_K_series: List[List[float]] = []    # [[t, I_K], ...]
        self._I_L_series: List[List[float]] = []    # [[t, I_L], ...]
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return {"state"}

    def outputs(self) -> Set[str]:
        return {"hh_state_summary"}

    def reset(self) -> None:
        self._v_series = []
        self._m_series = []
        self._h_series = []
        self._n_series = []
        self._I_Na_series = []
        self._I_K_series = []
        self._I_L_series = []
        self._outputs = {}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get("state")
        if signal is None:
            return

        payload = signal.value or {}
        t = float(payload.get("t", signal.time))
        indices = payload.get("indices", [])

        # Find the position of our neuron in the sampled indices
        if self.neuron_index not in indices:
            # If the exact index isn't there, use the first available
            if not indices:
                return
            pos = 0
        else:
            pos = indices.index(self.neuron_index)

        # Extract values for this neuron
        v_vals = payload.get("v", [])
        m_vals = payload.get("m", [])
        h_vals = payload.get("h", [])
        n_vals = payload.get("n_gate", [])
        I_Na_vals = payload.get("I_Na", [])
        I_K_vals = payload.get("I_K", [])
        I_L_vals = payload.get("I_L", [])

        if pos < len(v_vals):
            self._v_series.append([t, float(v_vals[pos])])
        if pos < len(m_vals):
            self._m_series.append([t, float(m_vals[pos])])
        if pos < len(h_vals):
            self._h_series.append([t, float(h_vals[pos])])
        if pos < len(n_vals):
            self._n_series.append([t, float(n_vals[pos])])
        if pos < len(I_Na_vals):
            self._I_Na_series.append([t, float(I_Na_vals[pos])])
        if pos < len(I_K_vals):
            self._I_K_series.append([t, float(I_K_vals[pos])])
        if pos < len(I_L_vals):
            self._I_L_series.append([t, float(I_L_vals[pos])])

        # Trim if over limit
        for series in [self._v_series, self._m_series, self._h_series,
                       self._n_series, self._I_Na_series, self._I_K_series,
                       self._I_L_series]:
            if len(series) > self.max_points:
                del series[:-self.max_points]

    def advance_to(self, t: float) -> None:
        # Emit a compact summary for persistence / downstream analysis.
        latest = {
            "V_mV": self._v_series[-1][1] if self._v_series else None,
            "m": self._m_series[-1][1] if self._m_series else None,
            "h": self._h_series[-1][1] if self._h_series else None,
            "n": self._n_series[-1][1] if self._n_series else None,
            "I_Na": self._I_Na_series[-1][1] if self._I_Na_series else None,
            "I_K": self._I_K_series[-1][1] if self._I_K_series else None,
            "I_L": self._I_L_series[-1][1] if self._I_L_series else None,
        }

        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "hh_state_summary": BioSignal(
                source=source,
                name="hh_state_summary",
                value={
                    "t": float(t),
                    "neuron_index": int(self.neuron_index),
                    "latest": latest,
                },
                time=float(t),
                metadata=SignalMetadata(description="Hodgkin-Huxley state monitor summary", kind="state"),
            )
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def visualize(self) -> Optional[List["VisualSpec"]]:
        """Generate 3-panel HH state visualization."""
        if not self._v_series:
            return None

        panels = []

        # Panel 1: Membrane potential
        panels.append({
            "render": "timeseries",
            "data": {
                "series": [
                    {"name": f"V (Neuron {self.neuron_index})", "points": list(self._v_series)},
                ],
                "title": "Membrane Potential (mV)",
            },
            "description": (
                "Membrane potential of the Hodgkin-Huxley neuron. "
                "Action potentials show rapid depolarization (~+40mV) from Na+ influx, "
                "followed by repolarization and afterhyperpolarization from K+ efflux."
            ),
        })

        # Panel 2: Gating variables
        if self._m_series:
            gate_series = [
                {"name": "m (Na+ activation)", "points": list(self._m_series)},
            ]
            if self._h_series:
                gate_series.append({"name": "h (Na+ inactivation)", "points": list(self._h_series)})
            if self._n_series:
                gate_series.append({"name": "n (K+ activation)", "points": list(self._n_series)})

            panels.append({
                "render": "timeseries",
                "data": {
                    "series": gate_series,
                    "title": "Gating Variables (0-1)",
                },
                "description": (
                    "HH gating variables controlling ion channel conductance. "
                    "m: Na+ activation (fast, opens during spike). "
                    "h: Na+ inactivation (slower, closes after spike). "
                    "n: K+ activation (slow, opens during repolarization). "
                    "The interplay of these gates produces the action potential waveform."
                ),
            })

        # Panel 3: Ionic currents
        if self._I_Na_series:
            current_series = [
                {"name": "I_Na", "points": list(self._I_Na_series)},
            ]
            if self._I_K_series:
                current_series.append({"name": "I_K", "points": list(self._I_K_series)})
            if self._I_L_series:
                current_series.append({"name": "I_L", "points": list(self._I_L_series)})

            panels.append({
                "render": "timeseries",
                "data": {
                    "series": current_series,
                    "title": "Ionic Currents (\u00b5A/cm\u00b2)",
                },
                "description": (
                    "Ionic currents flowing through the membrane. "
                    "I_Na: Large inward (negative) current during depolarization. "
                    "I_K: Outward current during repolarization, slightly delayed. "
                    "I_L: Small leak current providing baseline conductance."
                ),
            })

        return panels
