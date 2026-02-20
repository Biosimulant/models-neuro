# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Hodgkin-Huxley conductance-based spiking neuron model (1952).

Implements the classic squid giant axon model with voltage-gated Na+, K+,
and leak conductances. The membrane potential is governed by:

    C_m * dV/dt = I_ext - g_Na*m^3*h*(V - E_Na) - g_K*n^4*(V - E_K) - g_L*(V - E_L)

where m, h, n are gating variables obeying first-order kinetics:

    dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m
    dh/dt = alpha_h(V)*(1-h) - beta_h(V)*h
    dn/dt = alpha_n(V)*(1-n) - beta_n(V)*n

Reference:
    Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of membrane
    current and its application to conduction and excitation in nerve.
    J. Physiol., 117(4), 500-544.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim import BioWorld

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


def _safe_exp(x: float) -> float:
    """Compute exp(x) with overflow protection."""
    if x > 500.0:
        return math.exp(500.0)
    if x < -500.0:
        return 0.0
    return math.exp(x)


class HodgkinHuxleyPopulation(BioModule):
    """A population of Hodgkin-Huxley conductance-based spiking neurons.

    Receives `current` signals and emits `spikes` and `state` signals.
    State output includes voltage, gating variables, and ionic currents,
    making it compatible with both StateMonitor and HHStateMonitor.

    Parameters:
        n: Number of neurons.
        C_m: Membrane capacitance (uF/cm^2).
        g_Na: Maximum sodium conductance (mS/cm^2).
        g_K: Maximum potassium conductance (mS/cm^2).
        g_L: Leak conductance (mS/cm^2).
        E_Na: Sodium reversal potential (mV).
        E_K: Potassium reversal potential (mV).
        E_L: Leak reversal potential (mV).
        V_init: Initial membrane potential (mV).
        spike_threshold: Voltage threshold for spike detection (mV).
        I_bias: Constant bias current added to all neurons (uA/cm^2).
        sample_indices: Indices of neurons to include in state output.

    Emits:
        spikes: [int, ...] neuron indices that spiked this step
        state: {"t", "indices", "v", "m", "h", "n_gate", "I_Na", "I_K", "I_L"}
    """

    def __init__(
        self,
        n: int = 1,
        C_m: float = 1.0,
        g_Na: float = 120.0,
        g_K: float = 36.0,
        g_L: float = 0.3,
        E_Na: float = 50.0,
        E_K: float = -77.0,
        E_L: float = -54.387,
        V_init: float = -65.0,
        spike_threshold: float = 0.0,
        I_bias: float = 0.0,
        sample_indices: Optional[List[int]] = None,
        min_dt: float = 0.0001,
    ) -> None:
        self.min_dt = min_dt
        self.n = n
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.V_init = V_init
        self.spike_threshold = spike_threshold
        self.I_bias = I_bias

        # Sample indices for state output
        default_samples = list(range(min(5, n)))
        self.sample_indices = sample_indices if sample_indices is not None else default_samples

        # State arrays
        self._V: List[float] = []
        self._m: List[float] = []
        self._h: List[float] = []
        self._n: List[float] = []
        self._I_ext: List[float] = []
        self._time: float = 0.0
        self._last_dt: float = 0.0001

        # History for visualization (sampled neurons)
        self._v_history: Dict[int, List[List[float]]] = {}
        self._gate_history: Dict[int, Dict[str, List[List[float]]]] = {}
        self._outputs: Dict[str, BioSignal] = {}

        self._init_state()

    def _init_state(self) -> None:
        """Initialize neuron state arrays with steady-state gating variables."""
        V0 = self.V_init

        # Compute steady-state gating variables at resting potential
        am = self.alpha_m(V0)
        bm = self.beta_m(V0)
        ah = self.alpha_h(V0)
        bh = self.beta_h(V0)
        an = self.alpha_n(V0)
        bn = self.beta_n(V0)

        m_inf = am / (am + bm) if (am + bm) > 0 else 0.0
        h_inf = ah / (ah + bh) if (ah + bh) > 0 else 1.0
        n_inf = an / (an + bn) if (an + bn) > 0 else 0.0

        self._V = [V0] * self.n
        self._m = [m_inf] * self.n
        self._h = [h_inf] * self.n
        self._n = [n_inf] * self.n
        self._I_ext = [0.0] * self.n
        self._time = 0.0
        self._v_history = {}
        self._gate_history = {}

    @staticmethod
    def alpha_m(V: float) -> float:
        """Na+ activation rate: alpha_m(V) = 0.1*(V+40) / (1 - exp(-(V+40)/10))"""
        x = V + 40.0
        if abs(x) < 1e-6:
            # L'Hopital limit at V = -40: alpha_m -> 1.0
            return 1.0
        return 0.1 * x / (1.0 - _safe_exp(-x / 10.0))

    @staticmethod
    def beta_m(V: float) -> float:
        """Na+ activation rate: beta_m(V) = 4.0 * exp(-(V+65)/18)"""
        return 4.0 * _safe_exp(-(V + 65.0) / 18.0)

    @staticmethod
    def alpha_h(V: float) -> float:
        """Na+ inactivation rate: alpha_h(V) = 0.07 * exp(-(V+65)/20)"""
        return 0.07 * _safe_exp(-(V + 65.0) / 20.0)

    @staticmethod
    def beta_h(V: float) -> float:
        """Na+ inactivation rate: beta_h(V) = 1 / (1 + exp(-(V+35)/10))"""
        return 1.0 / (1.0 + _safe_exp(-(V + 35.0) / 10.0))

    @staticmethod
    def alpha_n(V: float) -> float:
        """K+ activation rate: alpha_n(V) = 0.01*(V+55) / (1 - exp(-(V+55)/10))"""
        x = V + 55.0
        if abs(x) < 1e-6:
            # L'Hopital limit at V = -55: alpha_n -> 0.1
            return 0.1
        return 0.01 * x / (1.0 - _safe_exp(-x / 10.0))

    @staticmethod
    def beta_n(V: float) -> float:
        """K+ activation rate: beta_n(V) = 0.125 * exp(-(V+65)/80)"""
        return 0.125 * _safe_exp(-(V + 65.0) / 80.0)

    def inputs(self) -> Set[str]:
        return {"current"}

    def outputs(self) -> Set[str]:
        return {"spikes", "state"}

    def reset(self) -> None:
        """Reset state for a new simulation run."""
        self._init_state()

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get("current")
        if signal is None:
            return
        I = signal.value
        # Current is a continuous signal — replace (not accumulate) and hold
        # until the next set_inputs call.  This ensures that when the HH model
        # advances faster than the current source (different min_dt), the
        # injected current persists between source emissions.
        if isinstance(I, (int, float)):
            self._I_ext = [float(I)] * self.n
        elif isinstance(I, (list, tuple)):
            self._I_ext = [
                float(I[i]) if i < len(I) else 0.0
                for i in range(self.n)
            ]

    def advance_to(self, t: float) -> None:
        dt = t - self._time if t > self._time else self._last_dt
        self._last_dt = dt
        self._time = t

        # Convert dt from seconds to milliseconds (HH equations use ms timescale)
        dt_ms = dt * 1000.0

        # Sub-step at max 0.025 ms for numerical stability (HH is stiffer than Izh)
        n_substeps = max(1, int(dt_ms / 0.025))
        sub_dt = dt_ms / n_substeps

        spiked_ids: List[int] = []

        for i in range(self.n):
            V = self._V[i]
            m = self._m[i]
            h = self._h[i]
            n_gate = self._n[i]
            I_total = self._I_ext[i] + self.I_bias

            V_prev = V
            spiked_this_step = False

            for _ in range(n_substeps):
                # Rate functions at current voltage
                am = self.alpha_m(V)
                bm = self.beta_m(V)
                ah = self.alpha_h(V)
                bh = self.beta_h(V)
                an = self.alpha_n(V)
                bn = self.beta_n(V)

                # Ionic currents
                I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
                I_K = self.g_K * (n_gate ** 4) * (V - self.E_K)
                I_L = self.g_L * (V - self.E_L)

                # Membrane equation: C_m * dV/dt = I_ext - I_Na - I_K - I_L
                dV = (I_total - I_Na - I_K - I_L) / self.C_m * sub_dt

                # Gating variable equations
                dm = (am * (1.0 - m) - bm * m) * sub_dt
                dh = (ah * (1.0 - h) - bh * h) * sub_dt
                dn = (an * (1.0 - n_gate) - bn * n_gate) * sub_dt

                # Forward Euler update
                V += dV
                m += dm
                h += dh
                n_gate += dn

                # Clamp gating variables to [0, 1]
                m = max(0.0, min(1.0, m))
                h = max(0.0, min(1.0, h))
                n_gate = max(0.0, min(1.0, n_gate))

            # Spike detection: upward crossing of threshold
            if V_prev < self.spike_threshold <= V and not spiked_this_step:
                spiked_this_step = True
                if i not in spiked_ids:
                    spiked_ids.append(i)

            self._V[i] = V
            self._m[i] = m
            self._h[i] = h
            self._n[i] = n_gate

        # Note: _I_ext is NOT cleared — it holds the last received value
        # until the next set_inputs call (continuous current semantics).

        # Compute ionic currents for state output (at final voltage)
        state_indices = [i for i in self.sample_indices if i < self.n]
        state_v = [self._V[i] for i in state_indices]
        state_m = [self._m[i] for i in state_indices]
        state_h = [self._h[i] for i in state_indices]
        state_n = [self._n[i] for i in state_indices]
        state_I_Na = [
            self.g_Na * (self._m[i] ** 3) * self._h[i] * (self._V[i] - self.E_Na)
            for i in state_indices
        ]
        state_I_K = [
            self.g_K * (self._n[i] ** 4) * (self._V[i] - self.E_K)
            for i in state_indices
        ]
        state_I_L = [
            self.g_L * (self._V[i] - self.E_L)
            for i in state_indices
        ]

        # Record history for visualization
        for idx, v_val in zip(state_indices, state_v):
            if idx not in self._v_history:
                self._v_history[idx] = []
            self._v_history[idx].append([t, v_val])

        source_name = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "spikes": BioSignal(
                source=source_name,
                name="spikes",
                value=spiked_ids,
                time=t,
                metadata=SignalMetadata(units=None, description="Spike events", kind="event"),
            ),
            "state": BioSignal(
                source=source_name,
                name="state",
                value={
                    "t": t,
                    "indices": state_indices,
                    "v": state_v,
                    "m": state_m,
                    "h": state_h,
                    "n_gate": state_n,
                    "I_Na": state_I_Na,
                    "I_K": state_I_K,
                    "I_L": state_I_L,
                },
                time=t,
                metadata=SignalMetadata(units="mV", description="HH neuron state", kind="state"),
            ),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self._time,
            "V": list(self._V),
            "m": list(self._m),
            "h": list(self._h),
            "n": list(self._n),
        }

    def visualize(self) -> Optional[List[Dict[str, Any]]]:
        """Return visualizations: V(t) traces and gating variables."""
        if not self._v_history:
            return None

        # Panel 1: Membrane potential traces
        v_series = []
        for idx in sorted(self._v_history.keys()):
            v_series.append({
                "name": f"Neuron {idx}",
                "points": self._v_history[idx],
            })

        panels = [
            {
                "render": "timeseries",
                "data": {
                    "series": v_series,
                    "title": f"HodgkinHuxley (n={self.n}) Membrane Potential",
                },
                "description": (
                    f"Membrane potential of {len(self._v_history)} sampled Hodgkin-Huxley neurons. "
                    f"Parameters: g_Na={self.g_Na}, g_K={self.g_K}, g_L={self.g_L} mS/cm\u00b2. "
                    "Action potentials show the characteristic rapid depolarization (Na+ influx) "
                    "followed by repolarization (K+ efflux) and a brief afterhyperpolarization."
                ),
            },
        ]

        return panels
