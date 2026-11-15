"""Phase 10 outage tests: the live (spin-up) path at 2024 origins with GOES, MRMS, HRRR and the KMIA
observations removed one at a time (and together). Each run must give a forecast or ABSTAIN, never
crash. 2024 is the validation year, so the cost of each outage is scored against the 1-minute truth.
Nothing is fitted or tuned. Writes out/phase10/outage_tests.{csv,md}."""
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from kmia2h.config import CFG, out

CONFIGS = [(), ("goes",), ("mrms",), ("hrrr",), ("kmia",), ("goes", "mrms", "hrrr"), ("goes", "mrms", "hrrr", "kmia")]
N_ORIGINS = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def one(args):
    o, outage = args
    from kmia2h.live import run as live
    t = time.time()
    try:
        f = live.forecast(live.Run(origin=o, mode="spinup", outage=outage))
    except Exception as e:
        return {"origin": o, "outage": "+".join(outage) or "none", "status": "CRASH", "error": traceback.format_exc()[-400:]}
    r = {"origin": o, "outage": "+".join(outage) or "none", "status": f["status"], "seconds": round(time.time() - t, 1),
         "degraded": " / ".join(f.get("degraded", [])), "reason": f.get("reason", "")}
    if f["status"] == "OK":
        for side in ("high", "low"):
            r[f"{side}_bins"] = json.dumps(f[side]["bins"])
            r[f"{side}_mode"], r[f"{side}_pmode"], r[f"{side}_action"] = f[side]["mode"], f[side]["p_mode"], f[side]["action"]
    return r


def main():
    from kmia2h.backtest import windows
    w = windows.load("v1", "2024-01", "2024-12")
    truth = pd.DataFrame({"hi": w["y_hi"], "lo": w["y_lo"]}, index=w["origins"]).dropna()
    rng = np.random.default_rng(10)
    origins = pd.DatetimeIndex(sorted(rng.choice(truth.index, N_ORIGINS, replace=False)))
    jobs = [(o, c) for o in origins for c in CONFIGS]
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(one, jobs, chunksize=2))
    d = pd.DataFrame(rows)
    for side, col in (("high", "hi"), ("low", "lo")):
        y = truth[col].reindex(d.origin).to_numpy()
        ll = []
        for b, yy in zip(d.get(f"{side}_bins", pd.Series([None] * len(d))), y):
            if not isinstance(b, str):
                ll.append(np.nan); continue
            p = json.loads(b).get(str(int(yy)), 0.0)
            ll.append(-np.log(max(p, 1e-4)))
        d[f"{side}_logloss"] = ll
        d[f"{side}_hit"] = np.where(d[f"{side}_mode"].notna(), d[f"{side}_mode"] == y, np.nan)
    d.drop(columns=[c for c in d.columns if c.endswith("_bins")]).to_csv(out(10, "outage_tests.csv"), index=False)
    g = d.groupby("outage", sort=False).agg(runs=("status", "size"), ok=("status", lambda s: int((s == "OK").sum())),
                                            abstain=("status", lambda s: int((s == "ABSTAIN").sum())),
                                            crash=("status", lambda s: int((s == "CRASH").sum())),
                                            sec_median=("seconds", "median"), high_logloss=("high_logloss", "mean"),
                                            low_logloss=("low_logloss", "mean"), high_hit=("high_hit", "mean"),
                                            low_hit=("low_hit", "mean")).reset_index()
    # paired cost against no outage, on the origins where both gave a forecast
    base = d[d.outage == "none"].set_index("origin")
    for side in ("high", "low"):
        g[f"{side}_dlogloss_paired"] = [
            float((d[(d.outage == o_)].set_index("origin")[f"{side}_logloss"] - base[f"{side}_logloss"]).mean())
            for o_ in g.outage]
    deg = d[d.outage != "none"].groupby("outage").degraded.agg(lambda s: s.value_counts().index[0] if len(s) else "")
    lines = ["# Phase 10 — outage tests", "",
             f"Live (spin-up) path at {N_ORIGINS} random 2024 origins (validation year), each source removed for the "
             "60 min before the origin. Every run must end in a forecast or ABSTAIN. Log-loss is scored against the "
             "1-minute truth; the paired column is the mean change against the same origins with no outage. "
             "Nothing was tuned on these numbers.", "",
             g.to_markdown(index=False, floatfmt=".3f"), "", "Typical degradation notes:", ""]
    lines += [f"- **{k}**: {v}" for k, v in deg.items()]
    crashes = d[d.status == "CRASH"]
    lines += ["", f"Crashes: {len(crashes)}", ""]
    if len(crashes):
        lines += ["```", crashes.iloc[0].error, "```"]
    out(10, "outage_tests.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
