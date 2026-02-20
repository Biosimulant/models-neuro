from __future__ import annotations


def test_emits_spike_summary(biosim):
    from biosim.signals import BioSignal, SignalMetadata
    from src.spike_monitor import SpikeMonitor

    mon = SpikeMonitor(max_events=100, min_dt=0.001)
    mon.set_inputs(
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
    mon.advance_to(0.001)
    out = mon.get_outputs()
    assert "spike_summary" in out
    payload = out["spike_summary"].value
    assert payload["n_events"] >= 2

