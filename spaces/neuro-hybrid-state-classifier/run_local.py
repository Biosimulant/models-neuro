#!/usr/bin/env python3
"""Run neuro-hybrid-state-classifier locally without UI."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml


def _load_space() -> dict:
    return yaml.safe_load((Path(__file__).resolve().parent / "space.yaml").read_text(encoding="utf-8")) or {}


def _repo_root_map(current_repo_root: Path) -> dict[str, Path]:
    return {"Biosimulant/models-neuro": current_repo_root.resolve()}


def _resolve_model_manifest(repo_map: dict[str, Path], model_ref: dict) -> Path:
    repo_full_name = str(model_ref.get("repo") or model_ref.get("repo_full_name") or "").strip()
    manifest_rel = str(model_ref.get("manifest_path") or "").strip()
    if repo_full_name not in repo_map:
        raise RuntimeError(f"Unknown repo in model ref: {repo_full_name}")
    if not manifest_rel:
        raise RuntimeError("Missing manifest_path")
    return (repo_map[repo_full_name] / manifest_rel).resolve()


def _resolve_entrypoint(manifest_path: Path) -> tuple[str, str, dict]:
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    biosim = data.get("biosim") or {}
    ep = str(biosim.get("entrypoint") or "")
    if ":" not in ep:
        raise RuntimeError(f"Invalid entrypoint in {manifest_path}: {ep}")
    module_name, class_name = ep.split(":", 1)
    init_kwargs = dict(biosim.get("init_kwargs") or {})
    return module_name, class_name, init_kwargs


def _clear_module_cache(module_name: str) -> None:
    root = module_name.split(".", 1)[0]
    to_delete = [k for k in sys.modules if k == root or k.startswith(f"{root}.")]
    for k in to_delete:
        sys.modules.pop(k, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", default="auto")
    parser.add_argument("--tick-dt", default="auto")
    args = parser.parse_args()

    current_repo_root = Path(__file__).resolve().parents[2]
    monorepo_root = current_repo_root.parents[1]
    biosim_src = monorepo_root / "biosim" / "src"
    if biosim_src.exists():
        sys.path.insert(0, str(biosim_src))

    import biosim

    space = _load_space()
    repo_map = _repo_root_map(current_repo_root)
    world = biosim.BioWorld()
    builder = biosim.WiringBuilder(world)

    for model_ref in space.get("models", []):
        manifest_path = _resolve_model_manifest(repo_map, model_ref)
        model_dir = manifest_path.parent
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        module_name, class_name, init_kwargs = _resolve_entrypoint(manifest_path)
        kwargs = dict(init_kwargs)
        kwargs.update(dict(model_ref.get("parameters") or {}))
        for key, value in list(kwargs.items()):
            if isinstance(value, str) and (key.endswith("path") or key.endswith("_path")):
                path = Path(value)
                if not path.is_absolute():
                    kwargs[key] = str((model_dir / path).resolve())
        _clear_module_cache(module_name)
        importlib.invalidate_caches()
        cls = getattr(importlib.import_module(module_name), class_name)
        builder.add(model_ref["alias"], cls(**kwargs))

    for wire in space.get("wiring", []):
        builder.connect(wire["from"], wire.get("to", []))

    builder.apply()
    runtime = space.get("runtime", {})
    duration = runtime.get("duration", 10.0) if args.duration == "auto" else float(args.duration)
    tick_dt = runtime.get("tick_dt", 1.0) if args.tick_dt == "auto" else float(args.tick_dt)

    world.run(duration=float(duration), tick_dt=float(tick_dt))

    visuals = world.collect_visuals()
    print(f"Ran space '{space.get('title', 'neuro-hybrid-state-classifier')}'")
    print(f"Duration={duration}, tick_dt={tick_dt}")
    print(f"Modules={len(getattr(world, 'module_names', []))}, visuals={len(visuals)}")


if __name__ == "__main__":
    main()
