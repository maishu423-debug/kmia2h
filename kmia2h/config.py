"""Config access. CFG is a dict subclass with attribute access, so CFG.target.rounding and
CFG["target"]["rounding"] both work."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _wrap(x):
    if isinstance(x, dict):
        return AttrDict({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def load(path: Path | None = None) -> AttrDict:
    with open(path or ROOT / "config.yaml", encoding="utf8") as f:
        return _wrap(yaml.safe_load(f))


CFG = load()


def store(name: str) -> Path:
    p = ROOT / CFG.paths.store
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def path(kind: str, *parts: str) -> Path:
    p = ROOT / CFG.paths[kind]
    p.mkdir(parents=True, exist_ok=True)
    return p.joinpath(*parts)


def out(phase: str | int, name: str) -> Path:
    p = ROOT / CFG.paths.out / f"phase{phase}"
    p.mkdir(parents=True, exist_ok=True)
    return p / name
