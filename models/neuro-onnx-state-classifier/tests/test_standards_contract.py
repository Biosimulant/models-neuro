from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml


def _find_bsim_src(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        cand = parent / "biosim" / "src"
        if (cand / "biosim").is_dir():
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
    entry = data["biosim"]["entrypoint"]
    module_name, class_name = entry.split(":", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls


def test_instantiation():
    cls = _load_module_class()
    module = cls(session_factory=lambda _path: None)
    assert getattr(module, "min_dt", 0) > 0
    assert isinstance(module.inputs(), set)
    assert isinstance(module.outputs(), set)
    assert len(module.outputs()) > 0
