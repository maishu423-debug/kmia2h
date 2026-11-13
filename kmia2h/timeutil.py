"""Timestamp <-> int64 nanoseconds, independent of pandas' default resolution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_ns(idx) -> np.ndarray:
    return pd.DatetimeIndex(idx).as_unit("ns").asi8.copy()


def from_ns(x) -> pd.DatetimeIndex:
    x = np.asarray(x, np.int64)
    unit = "ns" if (x.size == 0 or np.abs(x).max() > 1e17) else "us"
    return pd.to_datetime(x, unit=unit, utc=True)
