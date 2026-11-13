"""QC flags on KMIA 1-minute data (SPEC.md Phase 2 step 4). Flags, never deletions.

Bitmask: 1 = |dT| > max_step_f in one minute with no rain nearby, 2 = identical for > flat_minutes
in daylight, 4 = outside [min_f, max_f]. Every consumer drops flagged readings (REBUILD.md).

Tightened (DECISIONS.md, phase 1): the step flag marks an isolated spike (the value differs by more
than max_step_f from both neighbours, on the same side), and the flat flag needs temperature AND
dewpoint unchanged. The literal rules flagged 2.6% of minutes, mostly real plateaus and real
rain-cooled drops, and lowered the CLI low match from 98.1% to 97.3%."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kmia2h.config import CFG

STEP, FLAT, RANGE = 1, 2, 4


def flags(d: pd.DataFrame) -> pd.Series:
    q = CFG.qc
    t = d["tmpf"]
    out = np.zeros(len(d), dtype=np.int8)
    out |= np.where((t < q.min_f) | (t > q.max_f), RANGE, 0).astype(np.int8)

    rain = (d["precip"].fillna(0) > 0).astype(float).rolling(21, center=True, min_periods=1).max() > 0
    tv = t.where(out == 0)
    s1, s2 = tv.diff(), tv.diff(-1)
    spike = (s1.abs() > q.max_step_f) & (s2.abs() > q.max_step_f) & (np.sign(s1) == np.sign(s2)) & ~rain
    out |= np.where(spike, STEP, 0).astype(np.int8)

    from kmia2h.physics import solar
    sol = solar.cached(d.index[0], d.index[-1])
    day = sol["is_day"].reindex(d.index).fillna(False).to_numpy(bool)
    td = d["dwpf"]
    run_id = ((tv != tv.shift()) | (td != td.shift())).cumsum()
    run_len = tv.groupby(run_id).transform("size").where(tv.notna(), 0)
    day_in_run = pd.Series(day, index=d.index).groupby(run_id).transform("sum")
    flat = (run_len > q.flat_minutes) & (day_in_run > q.flat_minutes)
    out |= np.where(flat, FLAT, 0).astype(np.int8)
    return pd.Series(out, index=d.index, name="qc")


def clean(d: pd.DataFrame) -> pd.Series:
    """tmpf with every excluded QC flag (CFG.qc.exclude_mask) set to NaN."""
    return d["tmpf"].where((d["qc"].astype(int) & int(CFG.qc.exclude_mask)) == 0)
