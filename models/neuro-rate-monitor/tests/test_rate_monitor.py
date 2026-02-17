from __future__ import annotations


def test_emits_rate_state(bsim):
    from bsim.signals import BioSignal, SignalMetadata
    from src.rate_monitor import RateMonitor

    mon = RateMonitor(window_size=0.05, n_neurons=10, min_dt=0.001)
    mon.set_inputs(
        {
            "spikes": BioSignal(
                source="neurons",
                name="spikes",
                value=[0, 1, 2],
                time=0.05,
                metadata=SignalMetadata(description="test", kind="event"),
            )
        }
    )
    mon.advance_to(0.05)
    out = mon.get_outputs()
    assert "rate_state" in out
    payload = out["rate_state"].value
    assert payload["n_neurons"] == 10
    assert payload["rate_hz"] >= 0.0

