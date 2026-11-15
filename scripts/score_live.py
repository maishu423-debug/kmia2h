"""Daily scoring job (SPEC.md Phase 10, step 8): score past live forecasts once the KMIA 1-minute
archive has landed (about 1-2 days after the fact) into out/live_scores.parquet.

  python scripts/score_live.py            score every unscored live forecast older than --min-age-h
  python scripts/score_live.py --all      also score replay / spinup runs (testing)

Truth is the settlement rule on the NCEI 1-minute data (half_up of the rolling 5-minute mean, QC
exclusions as everywhere else). A forecast whose window is not yet fully in the archive is left for the
next run."""
import argparse
import io
import json

import numpy as np
import pandas as pd

from kmia2h.config import CFG, ROOT

LIVE = ROOT / CFG.paths.out / "live"
SCORES = ROOT / CFG.paths.out / "live_scores.parquet"


def fetch_1min(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    from kmia2h.data import iem
    from kmia2h.data.fetch_iem_1min import VARS
    from kmia2h.data.qc import flags
    params = [("station", CFG.iem_station), *[("vars", v) for v in VARS], ("sample", "1min"), ("what", "download"),
              ("tz", "UTC"), ("sts", f"{lo:%Y-%m-%dT%H:%MZ}"), ("ets", f"{hi:%Y-%m-%dT%H:%MZ}")]
    d = pd.read_csv(io.StringIO(iem._get(CFG.iem.base + "/cgi-bin/request/asos1min.py", params)),
                    na_values=["M", ""], low_memory=False)
    d = d.rename(columns={"valid(UTC)": "ts_utc"})
    d["ts_utc"] = pd.to_datetime(d["ts_utc"], utc=True)
    d = d.drop_duplicates("ts_utc").set_index("ts_utc").sort_index()
    for v in VARS:
        d[v] = pd.to_numeric(d[v], errors="coerce")
    d = d.reindex(pd.date_range(lo, hi, freq="min", tz="UTC"))
    d["qc"] = flags(d)
    return d


def settled(t1: pd.Series, a: pd.Timestamp, b: pd.Timestamp):
    """Settled high / low over minutes a..b (inclusive), None if > 5 minutes are missing."""
    from kmia2h.data.qc import clean
    from kmia2h.target import round_rule, rolling_mean
    lead = CFG.target.lead_min
    v = clean(t1).reindex(pd.date_range(a - pd.Timedelta(minutes=lead), b, freq="min", tz="UTC")).to_numpy(float)
    if np.isnan(v[lead:]).sum() > CFG.target.max_missing_min:
        return None
    r = round_rule(rolling_mean(v), CFG.target.rounding)[lead:]
    return int(np.nanmax(r)), int(np.nanmin(r))


def score_bins(bins: dict, y: int) -> tuple[float, bool]:
    p = bins.get(str(y), 0.0)
    mode = max(bins, key=bins.get)
    return float(-np.log(max(p, 1e-4))), int(mode) == y


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-age-h", type=float, default=30.0)
    a = ap.parse_args(argv)
    f = LIVE / "runs.jsonl"
    if not f.exists():
        print("no runs logged")
        return
    runs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    done = pd.read_parquet(SCORES) if SCORES.exists() else pd.DataFrame(columns=["origin_utc", "run_utc"])
    seen = set(zip(done.origin_utc, done.run_utc))
    now = pd.Timestamp.now(tz="UTC")
    todo = [r for r in runs if r.get("status") == "OK" and (a.all or r.get("mode") == "live")
            and (r["origin_utc"], r["run_utc"]) not in seen
            and now - pd.Timestamp(r["origin_utc"]) > pd.Timedelta(hours=a.min_age_h)]
    if not todo:
        print("nothing to score")
        return
    rows = []
    H = pd.Timedelta(minutes=CFG.sim.horizon_min)
    for day, grp in pd.Series(todo).groupby([pd.Timestamp(r["origin_utc"]).floor("D") for r in todo]):
        lo = day - pd.Timedelta(days=1)
        t1 = fetch_1min(lo, day + pd.Timedelta(days=1, hours=3))
        for r in grp:
            o = pd.Timestamp(r["origin_utc"])
            s = settled(t1, o + pd.Timedelta(minutes=1), o + H)
            if s is None:
                continue
            row = {"origin_utc": r["origin_utc"], "run_utc": r["run_utc"], "mode": r["mode"], "high": s[0], "low": s[1]}
            for side, y in (("high", s[0]), ("low", s[1])):
                ll, hit = score_bins(r[side]["bins"], y)
                row.update({f"{side}_logloss": ll, f"{side}_hit": hit, f"{side}_action": r[side]["action"]})
            if "day" in r:
                dd = r["day"]
                sd = settled(t1, pd.Timestamp(dd["window_start_utc"]), pd.Timestamp(dd["covers_to_utc"]))
                if sd is not None:
                    for side, y in (("high", sd[0]), ("low", sd[1])):
                        ll, hit = score_bins(dd[side]["bins"], y)
                        row.update({f"day_{side}": y, f"day_{side}_logloss": ll, f"day_{side}_hit": hit})
            rows.append(row)
    new = pd.DataFrame(rows)
    allr = pd.concat([done, new], ignore_index=True) if len(done) else new
    allr.to_parquet(SCORES, index=False)
    print(f"scored {len(new)} forecasts; {len(allr)} in {SCORES.relative_to(ROOT)}")
    if len(new):
        print(new.filter(regex="logloss|hit").mean().round(3).to_string())


if __name__ == "__main__":
    main()
