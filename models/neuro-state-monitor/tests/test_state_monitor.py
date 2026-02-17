from __future__ import annotations


def test_emits_voltage_summary(bsim):
    from bsim.signals import BioSignal, SignalMetadata
    from src.state_monitor import StateMonitor

    mon = StateMonitor(max_points=10, min_dt=0.001)
    mon.set_inputs(
        {
            "state": BioSignal(
                source="neurons",
                name="state",
                value={"t": 0.0, "indices": [0, 1], "v": [-65.0, -64.0]},
                time=0.0,
                metadata=SignalMetadata(description="test", kind="state"),
            )
        }
    )
    mon.advance_to(0.001)
    out = mon.get_outputs()
    assert "voltage_summary" in out
    payload = out["voltage_summary"].value
    assert payload["latest_v_mV"][0] == -65.0

