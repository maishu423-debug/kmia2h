"""Phase 1: confirm the rounding / averaging rule against the CLI daily max and min."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kmia2h.config import CFG, out, store
from kmia2h.data.qc import clean
from kmia2h.target import RULES, rolling_mean, round_rule


def daily_table(t1: pd.Series) -> pd.DataFrame:
    """Per LST day: max/min of the unrounded rolling mean, raw max/min, coverage, longest gap."""
    lst = t1.index + pd.Timedelta(hours=CFG.tz_lst_offset_hours)
    r = pd.Series(rolling_mean(t1.to_numpy()), index=t1.index)
    g = pd.DataFrame({"t": t1.to_numpy(), "r": r.to_numpy(), "day": lst.date})
    na = g["t"].isna()
    run = na.groupby((~na).cumsum()).transform("sum")
    g["gap"] = np.where(na, run, 0)
    return g.groupby("day").agg(rmax=("r", "max"), rmin=("r", "min"), raw_max=("t", "max"), raw_min=("t", "min"),
                                cov=("t", lambda x: x.notna().mean()), max_gap=("gap", "max"))


def main(first="2020-01-01", last="2025-12-31"):
    k = pd.read_parquet(store("kmia_1min.parquet"))
    k = k.loc[pd.Timestamp(first, tz="UTC") - pd.Timedelta(hours=CFG.tz_lst_offset_hours):
              pd.Timestamp(last, tz="UTC") + pd.Timedelta(days=1)]
    k_raw_flag = k["qc"]
    t1 = clean(k)
    dt = daily_table(t1)
    dt_noqc = daily_table(k["tmpf"])
    cli = pd.read_parquet(store("cli.parquet"))
    cli = cli[cli.final].set_index("date")
    j = dt.join(cli[["high", "low"]], how="inner")
    j = j[(j.index >= pd.Timestamp(first).date()) & (j.index <= pd.Timestamp(last).date())]
    j = j[j["cov"] >= CFG.target.cli_coverage_min].dropna(subset=["high", "low", "rmax", "rmin"])
    rows = []
    for rule in RULES:
        hi, lo = round_rule(j.rmax, rule), round_rule(j.rmin, rule)
        rows.append({"rule": rule, "n": len(j), "high_match": float((hi == j.high).mean()),
                     "low_match": float((lo == j.low).mean())})
    rows.append({"rule": "raw_1min", "n": len(j), "high_match": float((j.raw_max == j.high).mean()),
                 "low_match": float((j.raw_min == j.low).mean())})
    jn = dt_noqc.loc[j.index]
    rows.append({"rule": "half_up_without_qc", "n": len(j),
                 "high_match": float((round_rule(jn.rmax, "half_up") == j.high).mean()),
                 "low_match": float((round_rule(jn.rmin, "half_up") == j.low).mean())})
    res = pd.DataFrame(rows)
    # Highest combined match; ties go to the earliest rule in RULES (half_up first).
    cand = res[res.rule.isin(RULES)].assign(s=lambda d: (d.high_match + d.low_match).round(9))
    best = cand[cand.s == cand.s.max()].iloc[0]
    rule = best.rule
    # Mismatches with a cause column.
    hi, lo = round_rule(j.rmax, rule), round_rule(j.rmin, rule)
    qc_days = pd.Series(k_raw_flag.to_numpy() > 0,
                        index=(k.index + pd.Timedelta(hours=CFG.tz_lst_offset_hours)).date).groupby(level=0).any()
    mm = []
    for kind, pred, obs in (("high", hi, j.high), ("low", lo, j.low)):
        bad = pred != obs
        for d in j.index[bad]:
            cause = "gap" if j.loc[d, "max_gap"] > CFG.target.max_interp_gap_min else (
                "sensor_flag" if qc_days.get(d, False) else "unexplained")
            mm.append({"date": d, "kind": kind, "predicted": pred[d], "cli": obs[d],
                       "unrounded": j.loc[d, "rmax" if kind == "high" else "rmin"],
                       "coverage": j.loc[d, "cov"], "max_gap": j.loc[d, "max_gap"], "cause": cause})
    mm = pd.DataFrame(mm)
    mm.to_csv(out(1, "mismatches.csv"), index=False)
    res.to_csv(out(1, "rule_comparison.csv"), index=False)
    print(res.to_string(index=False))
    print("best", rule)
    if len(mm):
        print(mm.groupby(["kind", "cause"]).size())
        print("mean |pred-cli|", (mm.predicted - mm.cli).abs().mean())
    return res, rule, mm


if __name__ == "__main__":
    main()
