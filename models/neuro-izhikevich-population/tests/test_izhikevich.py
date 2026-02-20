"""Tests for IzhikevichPopulation model."""
from __future__ import annotations


def test_emits_state(biosim):
    from src.izhikevich import IzhikevichPopulation

    world = biosim.BioWorld()
    neurons = IzhikevichPopulation(n=5, min_dt=0.001)
    world.add_biomodule("neurons", neurons)
    world.run(duration=0.002, tick_dt=0.001)
    outputs = world.get_outputs("neurons")
    assert "state" in outputs
