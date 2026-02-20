from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MODEL_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _paths():
    p = str(_MODEL_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def biosim(_paths):
    import biosim as _bsim

    return _bsim

