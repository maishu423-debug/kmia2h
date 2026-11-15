"""Kalshi climate day (midnight to midnight LST): compose the observed part of the day with the 2-hour
forecast (SPEC.md Phase 10, step 6).

A live run has no KMIA 1-minute data, so the settled extreme of the elapsed part is known only through
the METAR / HFMETAR reports. The error D = settled extreme - round(report extreme) over the elapsed
part is an empirical pmf, fitted on 2022-2023 (real reports only, no synthetic HFMETAR) by elapsed
length, and stored in params/elapsed_obs.json.

Day high H = max(H_obs, H_fc), so P(H <= k) = P(H_obs <= k) P(H_fc <= k); the low is the mirror
image. Treating the two parts as independent ignores that both contain the temperature at the origin;
the 2024 check in out/phase10/compose_2024.md measures what that costs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kmia2h.config import CFG, out, path, store
from kmia2h.target import round_rule, rolling_mean

RULE = CFG.target.rounding
LST = pd.Timedelta(hours=CFG.tz_lst_offset_hours)
BUCKETS_H = [0, 1, 3, 6, 12, 24.01]
SUPPORT = np.arange(-6, 7)        # D values kept in the pmf
PMF_FLOOR = 1e-4


def lst_midnight(t: pd.Timestamp) -> pd.Timestamp:
    """UTC time of the LST midnight that starts the climate day containing t."""
    return (t + LST).floor("D") - LST


def bucket_of(hours: float) -> int:
    return int(np.clip(np.searchsorted(BUCKETS_H, hours, side="right") - 1, 0, len(BUCKETS_H) - 2))


def report_extremes(kmia: pd.DataFrame, start: pd.Timestamp, origin: pd.Timestamp) -> tuple[float, float, int]:
    """Max and min of KMIA temperature reports valid in [start, origin] and available by origin."""
    k = kmia[(kmia.ts_utc >= start) & (kmia.ts_utc <= origin) & (kmia.available_at <= origin) & kmia.tmpf.notna()]
    if "synthetic" in k:
        k = k[~k.synthetic.astype(bool)]
    if k.empty:
        return np.nan, np.nan, 0
    return float(k.tmpf.max()), float(k.tmpf.min()), int(len(k))


def fit(first: str = "2022-01-01", last: str = "2023-12-31", step_min: int = 30) -> dict:
    """Empirical pmf of D per side and elapsed bucket on [first, last] (2022-2023 only)."""
    from kmia2h.data.qc import clean
    k1 = pd.read_parquet(store("kmia_1min.parquet"))
    t1 = clean(k1)
    obs = pd.read_parquet(store("obs.parquet"), filters=[("station", "==", CFG.station)])
    obs = obs[obs.tmpf.notna() & ~obs.synthetic.astype(bool)]
    obs = obs.astype({"ts_utc": "datetime64[ns, UTC]", "available_at": "datetime64[ns, UTC]"})
    days = pd.date_range(pd.Timestamp(first, tz="UTC"), pd.Timestamp(last, tz="UTC"), freq="D")
    counts = {s: np.zeros((len(BUCKETS_H) - 1, len(SUPPORT))) for s in ("high", "low")}
    n_used = 0
    lead = CFG.target.lead_min
    for d in days:
        ws = d - LST                                                     # LST midnight of that date
        idx = pd.date_range(ws - pd.Timedelta(minutes=lead), ws + pd.Timedelta(days=1), freq="min", inclusive="left")
        v = t1.reindex(idx).to_numpy(float)
        r = round_rule(rolling_mean(v), RULE)[lead:]                     # reported value at minutes ws..
        miss = np.isnan(v[lead:])
        kd = obs[(obs.ts_utc >= ws) & (obs.ts_utc < ws + pd.Timedelta(days=1))]
        if kd.empty:
            continue
        ts = kd.ts_utc.dt.tz_convert(None).to_numpy()
        av = kd.available_at.dt.tz_convert(None).to_numpy()
        tv = kd.tmpf.to_numpy(float)
        for m in range(step_min, 24 * 60 + 1, step_min):
            o = ws + pd.Timedelta(minutes=m)
            seg, mseg = r[: m + 1], miss[: m + 1]
            if mseg.sum() > max(5, 0.02 * len(mseg)) or np.isnan(seg).all():
                continue
            on = o.tz_convert(None).to_datetime64()
            sel = (ts <= on) & (av <= on)
            if not sel.any():
                continue
            hb = bucket_of(m / 60)
            for side, true_x, rep in (("high", np.nanmax(seg), tv[sel].max()), ("low", np.nanmin(seg), tv[sel].min())):
                dd = int(true_x - round_rule(rep, RULE))
                if SUPPORT[0] <= dd <= SUPPORT[-1]:
                    counts[side][hb, dd - SUPPORT[0]] += 1
            n_used += 1
    pm = {"support": SUPPORT.tolist(), "buckets_h": BUCKETS_H, "fit_period": [first, last], "n": n_used,
          "rule": "D = settled extreme of the elapsed part - round(report extreme)"}
    for side, c in counts.items():
        p = (c + PMF_FLOOR) / (c + PMF_FLOOR).sum(1, keepdims=True)
        pm[side] = p.tolist()
        pm[f"{side}_n"] = c.sum(1).astype(int).tolist()
    path("params", "elapsed_obs.json").write_text(json.dumps(pm, indent=1))
    return pm


def load() -> dict:
    return json.loads(path("params", "elapsed_obs.json").read_text())


def observed_pmf(pm: dict, side: str, report_x: float, hours: float) -> tuple[np.ndarray, np.ndarray]:
    """(bins, probs) of the settled extreme of the elapsed part."""
    sup = np.array(pm["support"])
    p = np.array(pm[side][bucket_of(hours)])
    return int(round_rule(report_x, RULE)) + sup, p


def combine(ks_obs, p_obs, ks_fc, p_fc, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Distribution of max (high) or min (low) of two independent integer variables."""
    lo, hi = min(ks_obs.min(), ks_fc.min()), max(ks_obs.max(), ks_fc.max())
    ks = np.arange(lo, hi + 1)
    a = np.zeros(len(ks)); a[ks_obs - lo] = p_obs
    b = np.zeros(len(ks)); b[ks_fc - lo] = p_fc
    a, b = a / a.sum(), b / b.sum()
    if side == "high":
        F = np.cumsum(a) * np.cumsum(b)
        p = np.diff(np.r_[0.0, F])
    else:
        S = np.cumsum(a[::-1])[::-1] * np.cumsum(b[::-1])[::-1]        # P(X >= k)
        p = -np.diff(np.r_[S, 0.0])
    p = np.clip(p, 0, None)
    return ks, p / p.sum()


def validate_2024(tag: str | None = None, name: str | None = None, use=None) -> pd.DataFrame:
    """Score the composed day-so-far distribution (window start = LST midnight, end = origin + 2 h,
    origins whose 2-hour window ends before LST midnight) on 2024 against the 1-minute truth.
    Compares: composed, forecast window only, observed part only."""
    from kmia2h.data.qc import clean
    from kmia2h.mos import pipeline
    fz = CFG.frozen
    tag, name, use = tag or fz.backtest_tag, name or fz.mos_name, use or tuple(fz.sources)
    pm = load()
    d, res = pipeline.apply_frozen(tag, "2024-01", "2024-12", name, use)
    g0, g1 = pd.Timestamp("2023-12-31", tz="UTC"), pd.Timestamp("2025-01-02", tz="UTC")
    grid = pd.date_range(g0, g1, freq="min", inclusive="left")
    tv = clean(pd.read_parquet(store("kmia_1min.parquet"))).reindex(grid).to_numpy(float)   # once, not per origin
    rep_all = round_rule(rolling_mean(tv), RULE)
    obs = pd.read_parquet(store("obs.parquet"), filters=[("station", "==", CFG.station)])
    obs = obs.astype({"ts_utc": "datetime64[ns, UTC]", "available_at": "datetime64[ns, UTC]"})
    obs = obs[(obs.ts_utc >= g0) & (obs.ts_utc < g1) & obs.tmpf.notna() & ~obs.synthetic.astype(bool)]
    o_ts = ((obs.ts_utc - g0).dt.total_seconds() // 60).to_numpy(int)
    o_av = ((obs.available_at - g0).dt.total_seconds() // 60).to_numpy(int)
    o_t = obs.tmpf.to_numpy(float)
    order = np.argsort(o_ts)
    o_ts, o_av, o_t = o_ts[order], o_av[order], o_t[order]
    lead, H = CFG.target.lead_min, CFG.sim.horizon_min
    rows = []
    for i, o in enumerate(pd.DatetimeIndex(d.origin)):
        ws = lst_midnight(o)
        el_h = (o - ws).total_seconds() / 3600
        if o + pd.Timedelta(minutes=H) > ws + pd.Timedelta(days=1) or el_h < 0.5:
            continue
        a = int((ws - g0).total_seconds() // 60)
        b = int((o - g0).total_seconds() // 60) + H
        if np.isnan(tv[a:b + 1]).sum() > 5:
            continue
        r = rep_all[a:b + 1]
        oi = int((o - g0).total_seconds() // 60)
        lo_, hi_ = np.searchsorted(o_ts, a), np.searchsorted(o_ts, oi, side="right")
        sel = o_av[lo_:hi_] <= oi
        if not sel.any():
            continue
        hx, lx = o_t[lo_:hi_][sel].max(), o_t[lo_:hi_][sel].min()
        for side, truth, rep in (("high", np.nanmax(r), hx), ("low", np.nanmin(r), lx)):
            rr = res[side]
            ks_fc = rr["base"][i] - pipeline.OFF + np.arange(2 * pipeline.OFF + 1)
            p_fc = rr["blend"][i]
            ks_o, p_o = observed_pmf(pm, side, rep, el_h)
            ks, p = combine(ks_o, p_o, ks_fc, p_fc, side)
            out_ = {"origin": o, "side": side, "elapsed_h": el_h, "truth": int(truth)}
            for lab, kk, pp in (("composed", ks, p), ("forecast_only", ks_fc, p_fc), ("observed_only", ks_o, p_o)):
                j = np.flatnonzero(kk == int(truth))
                pr = pp[j[0]] if len(j) else 0.0
                out_[f"{lab}_logloss"] = float(-np.log(max(pr, 1e-4)))
                out_[f"{lab}_hit"] = bool(kk[np.argmax(pp)] == int(truth))
            rows.append(out_)
    t = pd.DataFrame(rows)
    t["elapsed_bucket"] = pd.cut(t.elapsed_h, BUCKETS_H, right=False).astype(str)
    agg = t.groupby(["side", "elapsed_bucket"]).agg(n=("truth", "size"), **{
        f"{m}_{s}": (f"{m}_{s}", "mean") for m in ("composed", "forecast_only", "observed_only") for s in ("logloss", "hit")}).reset_index()
    tot = t.groupby("side").agg(n=("truth", "size"), **{
        f"{m}_{s}": (f"{m}_{s}", "mean") for m in ("composed", "forecast_only", "observed_only") for s in ("logloss", "hit")}).reset_index()
    tot.insert(1, "elapsed_bucket", "all")
    agg = pd.concat([tot, agg], ignore_index=True)
    lines = ["# Composition check on 2024 (elapsed-part pmf fitted on 2022-2023)", "",
             "Target: settled extreme over [LST midnight, origin + 2 h] from the 1-minute truth. Origins whose 2-hour "
             "window crosses LST midnight, with < 30 min elapsed, or with no reports, are left out.", "",
             agg.to_markdown(index=False, floatfmt=".3f"), ""]
    out(10, "compose_2024.md").write_text("\n".join(lines))
    return agg
