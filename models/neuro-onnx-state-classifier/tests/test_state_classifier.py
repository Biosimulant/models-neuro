from __future__ import annotations

import importlib
import sys
import types


class _Port:
    def __init__(self, name: str):
        self.name = name


class _FakeSession:
    def get_inputs(self):
        return [_Port("state_vector")]

    def get_outputs(self):
        return [_Port("state_probabilities")]

    def run(self, _output_names, _feeds):
        return [[[0.1, 0.2, 0.7]]]


def test_predicts_label_from_fake_session(biosim):
    from biosim.signals import BioSignal, SignalMetadata
    from src.state_classifier import OnnxStateClassifier

    mod = OnnxStateClassifier(session_factory=lambda _path: _FakeSession())
    mod.set_inputs(
        {
            "state_vector": BioSignal(
                source="adapter",
                name="state_vector",
                value=[-60.0, 0.2, 0.7, 0.35],
                time=0.001,
                metadata=SignalMetadata(kind="state"),
            )
        }
    )
    mod.advance_to(0.001)
    out = mod.get_outputs()

    assert out["state_probabilities"].value == [0.1, 0.2, 0.7]
    assert out["predicted_state"].value["label"] == "spiking"


def test_falls_back_when_biosim_lacks_shared_onnx_helper(monkeypatch):
    class FakeBioModule:
        min_dt = 0.001

    fake_biosim = types.ModuleType("biosim")
    fake_biosim.BioModule = FakeBioModule

    class FakeBioSignal:
        def __init__(self, source, name, value, time, metadata=None):
            self.source = source
            self.name = name
            self.value = value
            self.time = time
            self.metadata = metadata

    class FakeSignalMetadata:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    fake_signals = types.ModuleType("biosim.signals")
    fake_signals.BioSignal = FakeBioSignal
    fake_signals.SignalMetadata = FakeSignalMetadata

    monkeypatch.setitem(sys.modules, "biosim", fake_biosim)
    monkeypatch.setitem(sys.modules, "biosim.signals", fake_signals)
    sys.modules.pop("src.compat_onnx", None)
    sys.modules.pop("src.state_classifier", None)

    module = importlib.import_module("src.state_classifier")
    cls = module.OnnxStateClassifier
    instance = cls(session_factory=lambda _path: _FakeSession())

    assert instance.outputs() == {"state_probabilities", "predicted_state"}
