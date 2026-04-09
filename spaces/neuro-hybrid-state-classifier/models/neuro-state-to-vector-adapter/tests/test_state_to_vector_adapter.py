from __future__ import annotations

import pytest


def test_builds_feature_vector_from_state_signal(biosim):
    from biosim.signals import BioSignal, SignalMetadata
    from src.state_to_vector_adapter import StateToVectorAdapter

    mod = StateToVectorAdapter()
    mod.set_inputs(
        {
            "state": BioSignal(
                source="hh",
                name="state",
                value={
                    "t": 0.001,
                    "v": [-65.0, -60.0],
                    "m": [0.1, 0.2],
                    "h": [0.8, 0.7],
                    "n_gate": [0.3, 0.4],
                },
                time=0.001,
                metadata=SignalMetadata(kind="state"),
            )
        }
    )
    mod.advance_to(0.001)
    out = mod.get_outputs()["state_vector"]
    assert out.value == pytest.approx([-62.5, 0.15, 0.75, 0.35])
    assert out.metadata.shape == (4,)
