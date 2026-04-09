"""Tests for HodgkinHuxleyPopulation model."""
from __future__ import annotations


DT = 0.0001  # 0.1ms — matches HH default min_dt


def _make_spike_catcher(biosim, spike_times, dt=DT):
    """Create a BioModule that records spike event times."""

    class Catcher(biosim.BioModule):
        def __init__(self):
            self.min_dt = dt

        def inputs(self):
            return {"spikes"}

        def set_inputs(self, signals):
            if "spikes" in signals and signals["spikes"].value:
                spike_times.append(signals["spikes"].time)

        def advance_to(self, t):
            pass

        def get_outputs(self):
            return {}

    return Catcher()


def test_produces_action_potentials(biosim):
    """HH neuron at I=10 uA/cm^2 should fire repeatedly."""
    from src.hodgkin_huxley import HodgkinHuxleyPopulation

    world = biosim.BioWorld()
    neuron = HodgkinHuxleyPopulation(n=1, I_bias=10.0, min_dt=DT)
    spike_times = []
    catcher = _make_spike_catcher(biosim, spike_times)

    world.add_biomodule("hh", neuron, priority=1)
    world.add_biomodule("catcher", catcher, priority=0)
    world.connect("hh.spikes", "catcher.spikes")
    world.run(duration=0.1, tick_dt=DT)

    assert len(spike_times) >= 5, f"Expected >=5 spikes, got {len(spike_times)}"


def test_subthreshold_no_sustained_firing(biosim):
    """HH neuron at I=2 uA/cm^2 should not produce sustained firing."""
    from src.hodgkin_huxley import HodgkinHuxleyPopulation

    world = biosim.BioWorld()
    neuron = HodgkinHuxleyPopulation(n=1, I_bias=2.0, min_dt=DT)
    spike_times = []
    catcher = _make_spike_catcher(biosim, spike_times)

    world.add_biomodule("hh", neuron, priority=1)
    world.add_biomodule("catcher", catcher, priority=0)
    world.connect("hh.spikes", "catcher.spikes")
    world.run(duration=0.1, tick_dt=DT)

    assert len(spike_times) <= 1, f"Expected <=1 spike, got {len(spike_times)}"


def test_state_output_keys(biosim):
    """HH state output should include voltage, gating vars, and ionic currents."""
    from src.hodgkin_huxley import HodgkinHuxleyPopulation

    world = biosim.BioWorld()
    neuron = HodgkinHuxleyPopulation(n=1, min_dt=DT)
    world.add_biomodule("hh", neuron)
    world.run(duration=0.001, tick_dt=DT)

    outputs = world.get_outputs("hh")
    assert "state" in outputs
    assert "spikes" in outputs

    state = outputs["state"].value
    expected_keys = {"t", "indices", "v", "m", "h", "n_gate", "I_Na", "I_K", "I_L"}
    assert expected_keys == set(state.keys())


def test_gating_variables_in_range(biosim):
    """Gating variables m, h, n should always be in [0, 1]."""
    from src.hodgkin_huxley import HodgkinHuxleyPopulation

    world = biosim.BioWorld()
    neuron = HodgkinHuxleyPopulation(n=1, I_bias=15.0, min_dt=DT)
    gate_values = []

    class Collector(biosim.BioModule):
        def __init__(self):
            self.min_dt = DT

        def inputs(self):
            return {"state"}

        def set_inputs(self, signals):
            if "state" in signals:
                sv = signals["state"].value
                gate_values.append((sv["m"][0], sv["h"][0], sv["n_gate"][0]))

        def advance_to(self, t):
            pass

        def get_outputs(self):
            return {}

    collector = Collector()
    world.add_biomodule("hh", neuron, priority=1)
    world.add_biomodule("col", collector, priority=0)
    world.connect("hh.state", "col.state")
    world.run(duration=0.05, tick_dt=DT)

    for m, h, n in gate_values:
        assert 0.0 <= m <= 1.0, f"m out of range: {m}"
        assert 0.0 <= h <= 1.0, f"h out of range: {h}"
        assert 0.0 <= n <= 1.0, f"n out of range: {n}"


def test_step_current_integration(biosim):
    """HH neuron driven by StepCurrent should fire during current injection."""
    from src.hodgkin_huxley import HodgkinHuxleyPopulation
    from src.step_current import StepCurrent

    world = biosim.BioWorld()
    stim = StepCurrent(I=0.0, schedule=[(0.01, 0.08, 10.0)], min_dt=DT)
    neuron = HodgkinHuxleyPopulation(n=1, min_dt=DT)
    spike_times = []
    catcher = _make_spike_catcher(biosim, spike_times)

    world.add_biomodule("stim", stim, priority=2)
    world.add_biomodule("hh", neuron, priority=1)
    world.add_biomodule("catcher", catcher, priority=0)
    world.connect("stim.current", "hh.current")
    world.connect("hh.spikes", "catcher.spikes")
    world.run(duration=0.1, tick_dt=DT)

    assert len(spike_times) >= 3, f"Expected >=3 spikes, got {len(spike_times)}"
    for t in spike_times:
        assert t >= 0.01, f"Spike at {t} before current onset"


def test_current_persists_across_ticks(biosim):
    """Current should persist even when StepCurrent has a slower min_dt."""
    from src.hodgkin_huxley import HodgkinHuxleyPopulation
    from src.step_current import StepCurrent

    world = biosim.BioWorld()
    stim = StepCurrent(I=10.0, min_dt=0.001)  # 10x slower than HH
    neuron = HodgkinHuxleyPopulation(n=1, min_dt=DT)
    spike_times = []
    catcher = _make_spike_catcher(biosim, spike_times)

    world.add_biomodule("stim", stim, priority=2)
    world.add_biomodule("hh", neuron, priority=1)
    world.add_biomodule("catcher", catcher, priority=0)
    world.connect("stim.current", "hh.current")
    world.connect("hh.spikes", "catcher.spikes")
    world.run(duration=0.1, tick_dt=DT)

    assert len(spike_times) >= 5, (
        f"Expected >=5 spikes with mismatched min_dt, got {len(spike_times)}. "
        "Current may not be persisting between StepCurrent emissions."
    )
