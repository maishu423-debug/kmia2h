"""What the framework needs from modules that live in the main repo.

The framework never imports target.py, window.py, features.py, data/ or floor/. Callers pass an
object satisfying these Protocols. The module `kmia2h.target` satisfies `Target` as it stands, so
in the main repo the call is `simulate.run(..., ops=kmia2h.target)`.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Target(Protocol):
    """Settlement operations, as defined in SPEC.md Phase 1."""

    def settle(self, t1: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
        """t1: (..., 4 + w) one-minute degF for minutes w0-3 .. w0+w. Returns (high, low) over the
        last w minutes of the reported (rolling, rounded) temperature."""
        ...

    def round_rule(self, x, rule: str):
        """Round degF values under a named rule. NaN stays NaN."""
        ...

    def edge(self, x):
        """|frac(x) - 0.5| of an unrounded extreme."""
        ...


@runtime_checkable
class Window(Protocol):
    """The market window around an origin, in whole minutes (the role of window.py)."""

    def elapsed_min(self, origin: pd.Timestamp) -> int:
        """Minutes of the market window already over at origin. 0 if it starts at origin."""
        ...

    def remaining_min(self, origin: pd.Timestamp) -> int:
        """Minutes of the market window after origin."""
        ...


@runtime_checkable
class FeatureProvider(Protocol):
    """Point-in-time feature rows (the role of features.py)."""

    def features_at(self, origins: pd.DatetimeIndex) -> pd.DataFrame:
        """One row per origin, using only rows with available_at <= origin."""
        ...
