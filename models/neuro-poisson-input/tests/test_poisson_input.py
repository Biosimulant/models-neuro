"""Tests for PoissonInput model."""
from __future__ import annotations


def test_emits_spikes(biosim):
    from src.poisson_input import PoissonInput

    world = biosim.BioWorld()
    poisson = PoissonInput(n=10, rate_hz=100.0, seed=42, min_dt=0.001)
    captured = []

    class SpikeCatcher(biosim.BioModule):
        def __init__(self):
            self.min_dt = 0.001

        def inputs(self):
            return {"spikes"}

        def set_inputs(self, signals):
            if "spikes" in signals:
                captured.append(signals["spikes"].value)

        def advance_to(self, t: float) -> None:
            return

        def get_outputs(self):
            return {}

    catcher = SpikeCatcher()
    world.add_biomodule("poisson", poisson, priority=1)
    world.add_biomodule("catcher", catcher, priority=0)
    world.connect("poisson.spikes", "catcher.spikes")

    world.run(duration=0.01, tick_dt=0.001)

    assert captured
    assert all(isinstance(v, list) for v in captured)


def test_deterministic_seed(biosim):
    from src.poisson_input import PoissonInput

    def run_once():
        world = biosim.BioWorld()
        poisson = PoissonInput(n=10, rate_hz=100.0, seed=123, min_dt=0.001)
        captured = []

        class SpikeCatcher(biosim.BioModule):
            def __init__(self):
                self.min_dt = 0.001

            def inputs(self):
                return {"spikes"}

            def set_inputs(self, signals):
                if "spikes" in signals:
                    captured.append(signals["spikes"].value)

            def advance_to(self, t: float) -> None:
                return

            def get_outputs(self):
                return {}

        catcher = SpikeCatcher()
        world.add_biomodule("poisson", poisson, priority=1)
        world.add_biomodule("catcher", catcher, priority=0)
        world.connect("poisson.spikes", "catcher.spikes")
        world.run(duration=0.01, tick_dt=0.001)
        return captured

    first = run_once()
    second = run_once()
    assert first == second
