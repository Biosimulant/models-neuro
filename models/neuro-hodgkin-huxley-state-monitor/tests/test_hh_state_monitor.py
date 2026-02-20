"""Tests for HHStateMonitor model."""
from __future__ import annotations


DT = 0.0001


def test_receives_state_and_produces_visuals(biosim):
    """HHStateMonitor should accumulate data and produce 3-panel visualization."""
    from src.hodgkin_huxley import HodgkinHuxleyPopulation
    from src.hodgkin_huxley_monitor import HHStateMonitor

    world = biosim.BioWorld()
    neuron = HodgkinHuxleyPopulation(n=1, I_bias=10.0, min_dt=DT)
    monitor = HHStateMonitor(max_points=5000, neuron_index=0, min_dt=DT)

    world.add_biomodule("hh", neuron, priority=1)
    world.add_biomodule("mon", monitor, priority=0)
    world.connect("hh.state", "mon.state")
    world.run(duration=0.01, tick_dt=DT)

    visuals = monitor.visualize()
    assert visuals is not None
    assert isinstance(visuals, list)
    assert len(visuals) == 3  # V(t), gates(t), currents(t)
    for panel in visuals:
        assert panel["render"] == "timeseries"
