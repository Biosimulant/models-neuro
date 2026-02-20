from __future__ import annotations


def test_emits_metrics(biosim):
    from biosim.signals import BioSignal, SignalMetadata
    from src.neuro_metrics import NeuroMetrics

    mod = NeuroMetrics(n_neurons=5, min_dt=0.001)
    mod.set_inputs(
        {
            "spikes": BioSignal(
                source="neurons",
                name="spikes",
                value=[0, 2],
                time=0.0,
                metadata=SignalMetadata(description="test", kind="event"),
            )
        }
    )
    mod.advance_to(0.001)
    out = mod.get_outputs()
    assert "metrics" in out
    payload = out["metrics"].value
    assert payload["n_neurons"] == 5
    assert payload["total_spikes"] >= 2

