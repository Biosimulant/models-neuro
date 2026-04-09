from __future__ import annotations


def test_emits_current(biosim):
    from src.step_current import StepCurrent

    mod = StepCurrent(I=5.0, min_dt=0.001)
    mod.advance_to(0.001)
    out = mod.get_outputs()
    assert "current" in out
    sig = out["current"]
    assert isinstance(sig.value, float)
    assert sig.value == 5.0

