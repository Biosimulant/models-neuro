from __future__ import annotations


def test_spikes_produce_current(biosim):
    from biosim.signals import BioSignal, SignalMetadata
    from src.exp_synapse import ExpSynapseCurrent

    syn = ExpSynapseCurrent(n_pre=3, n_post=4, p_connect=1.0, weight=2.0, tau=1.0, seed=1, min_dt=0.001)
    syn.set_inputs(
        {
            "spikes": BioSignal(
                source="pre",
                name="spikes",
                value=[0, 1],
                time=0.0,
                metadata=SignalMetadata(description="test spikes", kind="event"),
            )
        }
    )
    syn.advance_to(0.001)
    out = syn.get_outputs()
    assert "current" in out
    current = out["current"].value
    assert isinstance(current, list)
    assert len(current) == 4
    assert max(current) > 0.0

