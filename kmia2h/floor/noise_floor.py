"""kmia2h/floor/noise_floor.py — irreducible noise floor from KMIA history (SPEC.md Phase 3)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import optimize, stats

from kmia2h.config import CFG, out, path, store
from kmia2h.target import round_rule

H, RULE = CFG.sim.horizon_min, CFG.target.rounding
LEVELS = [['season', 'is_day', 'sky_class', 'wind'], ['season', 'is_day', 'sky_class'], ['season', 'is_day']]


def fwd(x: pd.Series, n: int, how: str) -> pd.Series:
    """Extreme of x over minutes t+1 .. t+n, aligned at t; NaN if any value is NaN."""
    r = x.iloc[::-1].rolling(n, min_periods=n)
    return (r.max() if how == 'max' else r.min()).iloc[::-1].shift(-1)


def fill_short(x: pd.Series, k: int = 3) -> pd.Series:
    na = x.isna()
    run = na.groupby((~na).cumsum()).transform('sum')
    return x.interpolate(limit_area='inside').where(~(na & (run > k)))


def floor_table(t1: pd.Series, precip: pd.Series, L: int, step_min: int = 10) -> pd.DataFrame:
    t1f = fill_short(t1, CFG.target.max_interp_gap_min)
    m = t1f.rolling(L, center=True, min_periods=L).mean()
    r, rm = t1f.rolling(5, min_periods=4).mean(), m.rolling(5, min_periods=5).mean()
    nmiss = t1.isna().astype(float).iloc[::-1].rolling(H).sum().iloc[::-1].shift(-1)
    d = pd.DataFrame({'Hr': fwd(r, H, 'max'), 'Lr': fwd(r, H, 'min'),
                      'Hm': fwd(rm, H, 'max'), 'Lm': fwd(rm, H, 'min'), 'nmiss': nmiss,
                      'rain_win': fwd((precip.fillna(0) > 0).astype(float), H, 'max') > 0})
    d = d[(d.index.minute % step_min == 0) & (d.nmiss <= CFG.target.max_missing_min)].dropna()
    d['eH'], d['eL'] = d.Hr - d.Hm, d.Lr - d.Lm
    d['hi'], d['lo'] = round_rule(d.Hr, RULE), round_rule(d.Lr, RULE)
    d['edge_H'], d['edge_L'] = (d.Hm % 1 - 0.5).abs(), (d.Lm % 1 - 0.5).abs()
    return d


def regime_keys(d: pd.DataFrame) -> dict[int, pd.Series]:
    return {i: d[lv].astype(str).agg('|'.join, axis=1) for i, lv in enumerate(LEVELS)}


def assign_regime(d: pd.DataFrame, train: pd.Series, min_n: int = 300) -> pd.Series:
    """Finest regime key with >= min_n training windows."""
    key = pd.Series(None, index=d.index, dtype=object)
    for lv in LEVELS:
        k = d[lv].astype(str).agg('|'.join, axis=1)
        ok = k.map(k[train].value_counts()).fillna(0) >= min_n
        key = key.where(key.notna() | ~ok, k)
    return key


def bin_probs(point: np.ndarray, resid_sorted: np.ndarray, offsets=range(-15, 16)):
    """P(round(point + e) = k) for k = floor(point) + offset, e ~ empirical residuals."""
    base = np.floor(point)
    ks = base[:, None] + np.array(list(offsets))[None, :]
    cdf = lambda x: np.searchsorted(resid_sorted, x, side='left') / len(resid_sorted)
    p = cdf(ks + 0.5 - point[:, None]) - cdf(ks - 0.5 - point[:, None])
    return ks, np.clip(p, 1e-4, None) / np.clip(p, 1e-4, None).sum(1, keepdims=True)


def score(d, point_col, resid_col, realised_col, key, train, test):
    out_ = []
    for g, rows in d[test].groupby(key[test]):
        res = np.sort(d.loc[train & (key == g), resid_col].values)
        ks, p = bin_probs(rows[point_col].values, res)
        hit = ks == rows[realised_col].values[:, None]
        out_.append(pd.DataFrame({'logloss': -np.log(np.maximum((p * hit).sum(1), 1e-4)),
                                  'mode_hit': ks[np.arange(len(ks)), p.argmax(1)] == rows[realised_col].values},
                                 index=rows.index))
    return pd.concat(out_)


def acov_fit(x: np.ndarray, maxlag: int = 90) -> dict:
    x = x - np.nanmean(x)
    c = np.array([np.nanmean(x[:len(x) - k] * x[k:]) for k in range(maxlag + 1)])
    k = np.arange(1, maxlag + 1)
    f = lambda k, af, tf, as_, ts: af * np.exp(-tf * k) + as_ * np.exp(-ts * k)
    (af, tf, as_, ts), _ = optimize.curve_fit(
        f, k, c[1:], p0=[c[1] * .6, .3, c[1] * .4, .04], bounds=(0, np.inf), maxfev=20000)
    if tf < ts: af, tf, as_, ts = as_, ts, af, tf
    return dict(A_f=af, theta_f=tf, A_s=as_, theta_s=ts, white=max(c[0] - af - as_, 0.0))


# ---------------------------------------------------------------------------------------------
# Driver

def load_t1() -> tuple[pd.Series, pd.Series]:
    from kmia2h.data.qc import clean
    k = pd.read_parquet(store("kmia_1min.parquet"))
    k = k.loc[:pd.Timestamp(CFG.periods.validation[1], tz="UTC") + pd.Timedelta(hours=3)]
    return clean(k), k["precip"]


def attach(d: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "is_day", "sky_class", "wind", "sector", "dd_bucket", "hour_lst", "t_last", "trend_30",
            "t_age_min"]
    return d.join(f[cols], how="inner")


def baseline_point(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Persistence plus 30-minute trend from live HFMETAR/METAR."""
    b = d.trend_30 / 30.0
    end = d.t_last + 60 * b
    return np.maximum(d.t_last, end), np.minimum(d.t_last, end)


def summarise(d, sc, prefix, train, test) -> dict:
    t = d[test]
    return {"n": int(test.sum())}


def gumbel(x: np.ndarray) -> dict:
    loc, scale = stats.gumbel_r.fit(x)
    return {"loc": float(loc), "scale": float(scale)}


def regime_params(d: pd.DataFrame, key: pd.Series, train: pd.Series) -> dict:
    outp = {}
    qs = np.arange(0.01, 1.0, 0.01)
    for g, rows in d[train].groupby(key[train]):
        e = {}
        for c, sign in (("eH", 1), ("eL", -1)):
            x = rows[c].to_numpy()
            e[c] = {"quantiles": [float(v) for v in np.quantile(x, qs)], "median": float(np.median(x)),
                    "std": float(x.std()), "mean": float(x.mean()), "n": int(len(x)),
                    "gumbel": gumbel(sign * x)}
        outp[g] = e
    return outp


def residual_acov(t1: pd.Series, L: int, f10: pd.DataFrame, maxlag: int) -> dict:
    """Per regime (all three hierarchy levels) autocovariance of X = t1 - m, fitted with two exponentials.
    Pairs never cross a gap, a day boundary or a regime change."""
    t1f = fill_short(t1, CFG.target.max_interp_gap_min)
    m = t1f.rolling(L, center=True, min_periods=L).mean()
    x = (t1f - m)
    tr = x.loc[:CFG.periods.floor_fit[1]]
    reg = f10[["season", "is_day", "sky_class", "wind"]].astype(str)
    reg = reg.reindex(tr.index, method="ffill")
    lst_day = (tr.index + pd.Timedelta(hours=CFG.tz_lst_offset_hours)).normalize()
    res = {}
    xv = tr.to_numpy()
    for lvl, cols in enumerate(LEVELS):
        k = reg[cols].agg("|".join, axis=1).to_numpy()
        codes, uniq = pd.factorize(k)
        seg = pd.Series(codes).astype(str) + "_" + pd.Series(lst_day.astype(str))
        seg = pd.factorize(seg)[0]
        nanseg = np.isnan(xv)
        seg = np.where(nanseg, -1, seg)
        # demean per regime
        mean = pd.Series(xv).groupby(codes).transform("mean").to_numpy()
        xd = xv - mean
        c = np.zeros((len(uniq), maxlag + 1))
        for lag in range(maxlag + 1):
            a, b = xd[:len(xd) - lag], xd[lag:]
            ok = (seg[:len(seg) - lag] == seg[lag:]) & (seg[lag:] >= 0)
            s = np.bincount(codes[:len(codes) - lag][ok], weights=(a * b)[ok], minlength=len(uniq))
            n = np.bincount(codes[:len(codes) - lag][ok], minlength=len(uniq))
            with np.errstate(invalid="ignore", divide="ignore"):
                c[:, lag] = s / n
            if lag == 0:
                n0 = n
        for i, g in enumerate(uniq):
            if n0[i] < 60 * 24 or g in res:
                continue
            ci = c[i]
            kk = np.arange(1, maxlag + 1)
            fn = lambda k, af, tf, as_, ts: af * np.exp(-tf * k) + as_ * np.exp(-ts * k)
            try:
                (af, tf, as_, ts), _ = optimize.curve_fit(fn, kk, ci[1:], p0=[ci[1] * .6, .3, ci[1] * .4, .04],
                                                          bounds=(0, np.inf), maxfev=20000)
            except Exception:
                continue
            if tf < ts:
                af, tf, as_, ts = as_, ts, af, tf
            res[g] = dict(A_f=float(af), theta_f=float(tf), A_s=float(as_), theta_s=float(ts),
                          white=float(max(ci[0] - af - as_, 0.0)), var0=float(ci[0]), n_minutes=int(n0[i]),
                          acov=[float(v) for v in ci])
    return res


def main():
    import time
    t_start = time.time()
    t1, precip = load_t1()
    f10 = pd.read_parquet(store("features_10m.parquet"))
    fit0, fit1 = CFG.periods.floor_fit
    v0, v1 = CFG.periods.validation
    lines = ["# Phase 3 — noise floor", ""]
    allrows = []
    for L in [CFG.floor.L_primary, *CFG.floor.L_sens]:
        t0 = time.time()
        d = floor_table(t1, precip, L, CFG.sim.origin_step_floor_min)
        took = time.time() - t0
        d = attach(d, f10)
        d = d[d.index <= pd.Timestamp(v1, tz="UTC") + pd.Timedelta(hours=22)]
        train = (d.index >= pd.Timestamp(fit0, tz="UTC")) & (d.index <= pd.Timestamp(fit1, tz="UTC") + pd.Timedelta(hours=23, minutes=59))
        test = (d.index >= pd.Timestamp(v0, tz="UTC")) & (d.index <= pd.Timestamp(v1, tz="UTC") + pd.Timedelta(hours=23, minutes=59))
        train, test = pd.Series(train, index=d.index), pd.Series(test, index=d.index)
        key = assign_regime(d, train, CFG.floor.min_regime_n)
        bh, bl = baseline_point(d)
        d["bH"], d["bL"] = bh, bl
        d["ebH"], d["ebL"] = d.Hr - d.bH, d.Lr - d.bL
        ok_base = d.bH.notna()
        sc = {}
        sc["oracle_H"] = score(d, "Hm", "eH", "hi", key, train, test)
        sc["oracle_L"] = score(d, "Lm", "eL", "lo", key, train, test)
        sc["base_H"] = score(d[ok_base], "bH", "ebH", "hi", key[ok_base], train[ok_base], test[ok_base])
        sc["base_L"] = score(d[ok_base], "bL", "ebL", "lo", key[ok_base], train[ok_base], test[ok_base])
        tt = d[test].copy()
        tt["naive_H"] = round_rule(tt.Hm, RULE) == tt.hi
        tt["naive_L"] = round_rule(tt.Lm, RULE) == tt.lo
        for k_, v in sc.items():
            tt[k_ + "_ll"] = v.logloss.reindex(tt.index)
            tt[k_ + "_hit"] = v.mode_hit.reindex(tt.index).astype(float)
        tt["group"] = tt.season + "-" + np.where(tt.is_day, "day", "night")
        rows = []
        for gname, g in [("all", tt), *[(n, x) for n, x in tt.groupby("group")]]:
            rows.append({"L": L, "group": gname, "n": len(g), "std_eH": g.eH.std(), "std_eL": g.eL.std(),
                         "naive_hit_H": g.naive_H.mean(), "oracle_hit_H": g.oracle_H_hit.mean(),
                         "oracle_ll_H": g.oracle_H_ll.mean(), "base_hit_H": g.base_H_hit.mean(),
                         "base_ll_H": g.base_H_ll.mean(),
                         "naive_hit_L": g.naive_L.mean(), "oracle_hit_L": g.oracle_L_hit.mean(),
                         "oracle_ll_L": g.oracle_L_ll.mean(), "base_hit_L": g.base_L_hit.mean(),
                         "base_ll_L": g.base_L_ll.mean()})
        s = pd.DataFrame(rows)
        allrows.append(s)
        eb = np.array(CFG.floor.edge_bins)
        tt["ebin_H"] = pd.cut(tt.edge_H, eb, include_lowest=True)
        tt["ebin_L"] = pd.cut(tt.edge_L, eb, include_lowest=True)
        edge_tab = pd.DataFrame({"oracle_hit_H": tt.groupby("ebin_H", observed=False).oracle_H_hit.mean(),
                                 "n_H": tt.groupby("ebin_H", observed=False).size(),
                                 "oracle_hit_L": tt.groupby("ebin_L", observed=False).oracle_L_hit.mean(),
                                 "n_L": tt.groupby("ebin_L", observed=False).size()})
        lines += [f"## L = {L} min  (floor_table over 2016-2024: {took:.0f} s)", "",
                  s.drop(columns="L").to_markdown(index=False, floatfmt=".3f"), "", "Hit rate by edge bin (2024):", "",
                  edge_tab.to_markdown(floatfmt=".3f"), ""]
        # Per-regime tables on 2024
        tt["regime"] = key[test]
        reg = tt.groupby("regime").agg(n=("eH", "size"), std_eH=("eH", "std"), std_eL=("eL", "std"),
                                       oracle_hit_H=("oracle_H_hit", "mean"), oracle_ll_H=("oracle_H_ll", "mean"),
                                       base_hit_H=("base_H_hit", "mean"), base_ll_H=("base_H_ll", "mean"),
                                       oracle_hit_L=("oracle_L_hit", "mean"), oracle_ll_L=("oracle_L_ll", "mean"),
                                       base_hit_L=("base_L_hit", "mean"), base_ll_L=("base_L_ll", "mean"))
        reg.to_csv(out(3, f"regime_L{L}.csv"))
        if L == CFG.floor.L_primary:
            d["regime"] = key
            d.to_parquet(store("floor_table_L21.parquet"))
            params = regime_params(d, key, train)
            path("params", "floor.json").write_text(json.dumps(params, indent=1))
            ac = residual_acov(t1, L, f10, CFG.floor.acov_maxlag)
            path("params", "residual_acov.json").write_text(json.dumps(ac, indent=1))
            primary = s
    out(3, "summary.md").write_text("\n".join(lines))
    pd.concat(allrows).to_csv(out(3, "summary.csv"), index=False)
    print("\n".join(lines))
    print(f"total {time.time() - t_start:.0f} s")
    return primary


if __name__ == "__main__":
    main()


def refit_positive_lags(ac: dict) -> dict:
    """Refit each stored autocovariance on lags 1 .. first zero crossing only.

    X = t1 - m with a centred 21-minute mean has a negative lobe near lags 5-10 that a sum of
    positive exponentials cannot follow; fitting all 90 lags gives degenerate parameters."""
    outp = {}
    for g, v in ac.items():
        c = np.asarray(v["acov"], float)
        neg = np.flatnonzero(c[1:] <= 0)
        k0 = int(neg[0]) if neg.size else len(c) - 1
        k0 = max(k0, 3)
        kk = np.arange(1, k0 + 1)
        f1 = lambda k, a, t: a * np.exp(-t * k)
        (a, t), _ = optimize.curve_fit(f1, kk, c[1:k0 + 1], p0=[c[1] * 2, 0.7], bounds=(0, np.inf), maxfev=20000)
        a = min(a, c[0])
        outp[g] = {**v, "A_f": float(a), "theta_f": float(t), "A_s": 0.0, "theta_s": float(t),
                   "white": float(max(c[0] - a, 0.0)), "fit_lags": int(k0),
                   "note": "single exponential on lags 1..first zero crossing; see REPORT.md Phase 3"}
    return outp


if __name__ == "__main__" and False:
    pass
