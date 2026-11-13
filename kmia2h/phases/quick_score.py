"""Physics-only 2024 score of a backtest tag: spread calibration fitted on 2022-2023, then bin
hit / log-loss on 2024, overall and by day/night x season. Used after each step-6 addition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kmia2h.mos import pipeline


def main(tag: str, first: str = "2022-01", last: str = "2024-12") -> pd.DataFrame:
    d, b = pipeline.assemble(tag, first, last)
    ok = (d.h_r.notna() & d.l_r.notna()).to_numpy()
    d = d[ok].reset_index(drop=True)
    tr = (d.origin < pd.Timestamp("2024-01-01", tz="UTC")).to_numpy()
    va = ~tr
    groups = pipeline.group_of(d.season, d.is_day)
    rows = []
    for side, S, real, tcol in (("high", b["max_u"][ok], "hi", "h_r"), ("low", b["min_u"][ok], "lo", "l_r")):
        base = np.floor(np.median(S, axis=1)).astype(int)
        realised = (d[real].to_numpy() - (base - pipeline.OFF)).astype(int)
        sc = pipeline.calib_spread(S[tr], base[tr], realised[tr], groups[tr], d[tcol].to_numpy()[tr])
        scale = np.array([sc[str(g)]["scale"] for g in groups])
        shift = np.array([sc[str(g)]["shift"] for g in groups])
        p_raw = pipeline.sample_probs(S, base, 1.0)
        p = pipeline.sample_probs(S, base, scale, shift)
        for name, P in (("raw", p_raw), ("calibrated", p)):
            rows.append({"tag": tag, "side": side, "model": name, "slice": "all", **pipeline.score(P[va], realised[va])})
            for g in np.unique(groups[va]):
                m = va & (groups == g)
                rows.append({"tag": tag, "side": side, "model": name, "slice": g, **pipeline.score(P[m], realised[m])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 200)
    print(main(sys.argv[1]).round(3).to_string(index=False))
