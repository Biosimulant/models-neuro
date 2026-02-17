from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml


def _find_bsim_src(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        cand = parent / "bsim" / "src"
        if (cand / "bsim").is_dir():
            return cand
    return None


def _ensure_paths() -> None:
    pack_root = Path(__file__).resolve().parents[1]
    if str(pack_root) not in sys.path:
        sys.path.insert(0, str(pack_root))

    bsim_src = _find_bsim_src(pack_root)
    if bsim_src is not None and str(bsim_src) not in sys.path:
        sys.path.insert(0, str(bsim_src))


def _load_module_class():
    _ensure_paths()
    manifest = Path(__file__).resolve().parents[1] / "model.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    entry = data["bsim"]["entrypoint"]
    module_name, class_name = entry.split(":", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls


def _make_instance_and_advance():
    cls = _load_module_class()
    module = cls()
    t = float(getattr(module, "min_dt", 1.0) or 1.0)
    if t <= 0:
        t = 1.0
    if hasattr(module, "inputs") and callable(module.inputs):
        ins = module.inputs()
        if ins and hasattr(module, "set_inputs") and callable(module.set_inputs):
            module.set_inputs({})
    module.advance_to(t)
    outputs = module.get_outputs()
    return module, outputs


def test_instantiation():
    cls = _load_module_class()
    module = cls()
    assert getattr(module, "min_dt", 0) > 0
    assert isinstance(module.inputs(), set)
    assert isinstance(module.outputs(), set)
    assert len(module.outputs()) > 0


def test_advance_produces_outputs():
    module, outputs = _make_instance_and_advance()
    assert isinstance(outputs, dict)
    for name in module.outputs():
        assert name in outputs


def test_output_keys_match():
    module, outputs = _make_instance_and_advance()
    assert set(outputs.keys()) == set(module.outputs())
