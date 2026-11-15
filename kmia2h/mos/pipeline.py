"""Phase 8 driver: spread calibration, MOS, conformal, blend, and 2024 scoring (SPEC.md Phase 8,
REBUILD steps 3-4). Fits on 2022-2023; conformal, blending and scores on 2024; 2025+ untouched."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from kmia2h.backtest import backtest, windows
from kmia2h.config import CFG, out, path, store
from kmia2h.floor import noise_floor
from kmia2h.forcing import model_cfg
from kmia2h.mos import conformal, predict, train
from kmia2h.target import round_rule

RULE = CFG.target.rounding
OFF = 15  # bins floor(median) - OFF .. + OFF


def save_model(model, side: str, name: str) -> None:
    """The compressed copy of a MOS model (joblib, zlib level 3) that the live bundle carries in git.
    The same pickle stream as the .pkl, compressed losslessly."""
    import joblib
    joblib.dump(model, path("models", f"mos_{side}_{name}.joblib"), compress=3)


def load_model(side: str, name: str):
    """models/mos_<side>_<name>.joblib (live bundle) if present, else the uncompressed .pkl."""
    jp = path("models", f"mos_{side}_{name}.joblib")
    if jp.exists():
        import joblib
        return joblib.load(jp)
    with open(path("models", f"mos_{side}_{name}.pkl"), "rb") as fh:
        return pickle.load(fh)


def group_of(season, is_day) -> np.ndarray:
    return conformal.group_key(season, np.asarray(is_day).astype(str) == "True")


def sample_probs(samples: np.ndarray, base: np.ndarray, scale: np.ndarray | float = 1.0, shift=0.0) -> np.ndarray:
    """(rows, 2*OFF+1) bin probabilities on bins base-OFF..base+OFF from unrounded extreme samples,
    after scaling the deviations from the median (spread calibration)."""
    med = np.median(samples, axis=1, keepdims=True)
    s = med + np.asarray(scale).reshape(-1, 1) * (samples - med) + np.asarray(shift).reshape(-1, 1)
    k = round_rule(s, RULE) - (base[:, None] - OFF)
    k = np.clip(k, 0, 2 * OFF).astype(int)
    p = np.zeros((len(samples), 2 * OFF + 1))
    np.add.at(p, (np.repeat(np.arange(len(samples)), samples.shape[1]), k.ravel()), 1.0)
    return p / samples.shape[1]


def score(p: np.ndarray, realised: np.ndarray, floor=1e-4) -> dict:
    ok = (realised >= 0) & (realised < p.shape[1])
    pr = np.where(ok, p[np.arange(len(p)), np.clip(realised, 0, p.shape[1] - 1)], 0.0)
    return {"hit": float(np.mean(p.argmax(1) == realised)), "logloss": float(np.mean(-np.log(np.maximum(pr, floor)))),
            "n": int(len(p))}


def calib_spread(samples, base, realised, groups, target, grid=np.round(np.arange(0.5, 3.01, 0.05), 2)) -> dict:
    """Per season x day/night: shift = median(target - sample median), then the deviation scale
    that minimises bin log-loss (REBUILD step 3, fitted on 2022-2023 only)."""
    out_ = {}
    sub = samples[:, ::4]
    for g in np.unique(groups):
        m = groups == g
        shift = float(np.nanmedian(target[m] - np.median(samples[m], axis=1)))
        best = min(grid, key=lambda s: score(sample_probs(sub[m], base[m], s, shift), realised[m])["logloss"])
        out_[str(g)] = {"scale": float(best), "shift": shift}
    return out_


def _table(name: str, tables: dict | None, columns=None) -> pd.DataFrame | None:
    key = name.replace(".parquet", "")
    if tables and key in tables:
        d = tables[key]
        return None if d is None else (d[columns].copy() if columns else d.copy())
    return pd.read_parquet(store(name), columns=columns) if store(name).exists() else None


def source_features(origins: pd.DatetimeIndex, use, tables: dict | None = None) -> pd.DataFrame:
    """GOES station cloud fraction, max upstream reflectivity within 40 km, HRRR 2 m T at window end,
    and avail_* flags (False where the source is missing or stale at the origin)."""
    from kmia2h.sources import MAX_GOES_AGE, MAX_MRMS_AGE
    o = pd.DataFrame({"origin": origins.astype("datetime64[ns, UTC]")})
    out = pd.DataFrame(index=origins, data={"cf_disc": np.nan, "max_dbz_up": np.nan, "hrrr_t2m_end": np.nan,
                                            "avail_goes": 0.0, "avail_mrms": 0.0, "avail_hrrr": 0.0})
    g = _table("goes_feat.parquet", tables, ["ts_utc", "available_at", "cf_disc"]) if "goes" in use else None
    if g is not None and len(g):
        g = g[g.cf_disc.notna()].astype({"available_at": "datetime64[ns, UTC]", "ts_utc": "datetime64[ns, UTC]"})
        m = pd.merge_asof(o, g.sort_values("available_at"), left_on="origin", right_on="available_at")
        fresh = (m.origin - m.ts_utc) <= MAX_GOES_AGE
        out["cf_disc"] = m.cf_disc.where(fresh).to_numpy()
        out["avail_goes"] = fresh.astype(float).to_numpy()
    mr = _table("mrms_feat.parquet", tables) if "mrms" in use else None
    if mr is not None and len(mr):
        cols = [c for c in mr.columns if c.startswith("maxdbz_s") and int(c[-1]) <= 2]
        mr["max_up"] = mr[cols].max(axis=1)
        mr = mr[["ts_utc", "available_at", "max_up"]].astype({"available_at": "datetime64[ns, UTC]", "ts_utc": "datetime64[ns, UTC]"})
        m = pd.merge_asof(o, mr.sort_values("available_at"), left_on="origin", right_on="available_at")
        fresh = (m.origin - m.ts_utc) <= MAX_MRMS_AGE
        out["max_dbz_up"] = m.max_up.where(fresh).to_numpy()
        out["avail_mrms"] = fresh.astype(float).to_numpy()
    h = _table("nwp_point.parquet", tables, ["cycle_utc", "valid_utc", "t2m"]) if "hrrr" in use else None
    if h is not None and len(h):
        h = h[h.t2m.notna()].astype({"cycle_utc": "datetime64[ns, UTC]", "valid_utc": "datetime64[ns, UTC]"})
        h["available_at"] = h.cycle_utc + pd.Timedelta(minutes=CFG.latency_min.hrrr)
        h["cyc_naive"] = h.cycle_utc.dt.tz_convert(None).astype("datetime64[ns]")
        cyc = {k: g for k, g in h.groupby("cyc_naive")}
        lat = pd.Timedelta(minutes=CFG.latency_min.hrrr)
        cycles = np.unique(h.cyc_naive.to_numpy())
        end = origins + pd.Timedelta(minutes=CFG.sim.horizon_min)
        vals, av = [], []
        for t, e in zip(origins, end):
            i = np.searchsorted(cycles, (t - lat).tz_convert(None).as_unit("ns").to_datetime64(), side="right") - 1
            if i < 0 or t - pd.Timestamp(cycles[i], tz="UTC") > pd.Timedelta(hours=6):
                vals.append(np.nan); av.append(0.0); continue
            g = cyc[pd.Timestamp(cycles[i])]
            vals.append(float(np.interp(e.as_unit("ns").value, g.valid_utc.astype("datetime64[ns, UTC]").astype("int64").to_numpy(), g.t2m.to_numpy())))
            av.append(1.0)
        out["hrrr_t2m_end"], out["avail_hrrr"] = vals, av
    return out


F30_COLS = ["t_last", "t_age_min", "trend_30", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "season", "is_day",
            "sky_class", "wind", "sector", "dd_bucket", "avail_neighbours", "avail_ndbc", "hour_lst", "rain_recent"]


def frame_from(b: dict, feats: pd.DataFrame, use=(), wt: pd.DataFrame | None = None,
               tables: dict | None = None) -> pd.DataFrame:
    """MOS input frame: backtest fields b (origin, edge_*, spread, ...), origin features (F30_COLS,
    indexed by origin), optional window truth wt and source features."""
    d = pd.DataFrame({"origin": b["origin"]}).set_index("origin", drop=False)
    if wt is not None:
        d = d.join(wt)
    d = d.join(feats[F30_COLS])
    for k in ("edge_high", "edge_low", "spread", "t_adv_mean", "c_mean", "mean_max", "mean_min", "ens_T_sd"):
        d[k] = b[k]
    d["ens_spread"] = d.spread
    d["t_adv_anom"] = d.t_adv_mean - d.t_last
    sf = source_features(pd.DatetimeIndex(d.index), use or (), tables)
    for c in ("cf_disc", "max_dbz_up", "hrrr_t2m_end", "avail_goes", "avail_mrms", "avail_hrrr"):
        d[c] = sf[c].to_numpy()
    for c, vals in (("season", ["wet", "dry"]), ("sky_class", ["clear", "sct", "bkn", "ovc"]),
                    ("sector", ["onshore", "offshore", "other"])):
        for v in vals:
            d[f"regime_{c}_{v}"] = (d[c] == v).astype(float)
    d["regime_is_day"] = d.is_day.astype(float)
    d["avail_neighbours"] = d.avail_neighbours.astype(float)
    d["avail_ndbc"] = d.avail_ndbc.astype(float)
    return d.reset_index(drop=True)


def assemble(tag: str, first: str, last: str, use=None) -> pd.DataFrame:
    b = backtest.load(tag, first, last)
    w = windows.load("v1", first, last)
    wt = pd.DataFrame({"h_r": w["y_h_r"], "l_r": w["y_l_r"], "hi": w["y_hi"], "lo": w["y_lo"]}, index=w["origins"])
    f30 = pd.read_parquet(store("features_30m.parquet"))
    return frame_from(b, f30, use, wt), b


def floor_baseline(d: pd.DataFrame) -> dict:
    """Phase 3 oracle and baseline on exactly these origins (2016-2023 residual fits)."""
    ft = pd.read_parquet(store("floor_table_L21.parquet"))
    tr = (ft.index < pd.Timestamp("2024-01-01", tz="UTC"))
    test = ft.index.isin(pd.DatetimeIndex(d.origin))
    trs, tes = pd.Series(tr, index=ft.index), pd.Series(test, index=ft.index)
    key = ft.regime
    res = {}
    for name, pc, rc, real in (("floor_high", "Hm", "eH", "hi"), ("floor_low", "Lm", "eL", "lo"),
                               ("baseline_high", "bH", "ebH", "hi"), ("baseline_low", "bL", "ebL", "lo")):
        ok = ft[pc].notna()
        s = noise_floor.score(ft[ok], pc, rc, real, key[ok], trs[ok], tes[ok])
        res[name] = s
    return res


def main(tag: str = "v2", reuse: bool = False, use=(), name: str | None = None):
    """name: label for saved models (default: the backtest tag)."""
    name = name or tag
    cfg = model_cfg()
    d, b = assemble(tag, "2022-01", "2024-12", use)
    ok = d.h_r.notna() & d.l_r.notna()
    d, mx, mn = d[ok].reset_index(drop=True), b["max_u"][ok.to_numpy()], b["min_u"][ok.to_numpy()]
    train_m = (d.origin < pd.Timestamp("2024-01-01", tz="UTC")).to_numpy()
    val_m = ~train_m
    groups = group_of(d.season, d.is_day)
    res = {}
    report = {}
    for side, S, real, tcol, qp in (("high", mx, "hi", "h_r", "q_max"), ("low", mn, "lo", "l_r", "q_min")):
        base = np.floor(np.median(S, axis=1)).astype(int)
        realised = (d[real].to_numpy() - (base - OFF)).astype(int)
        p_raw = sample_probs(S, base, 1.0)
        sc = calib_spread(S[train_m], base[train_m], realised[train_m], groups[train_m], d[tcol].to_numpy()[train_m])
        scale = np.array([sc.get(str(g), {"scale": 1.0})["scale"] for g in groups])
        shift = np.array([sc.get(str(g), {"shift": 0.0})["shift"] for g in groups])
        p_phys = sample_probs(S, base, scale, shift)
        # physics quantile features from the calibrated samples
        med = np.median(S, axis=1, keepdims=True)
        Sc = med + scale[:, None] * (S - med) + shift[:, None]
        for q in CFG.mos.physics_levels:
            d[f"{qp}_{q:02d}"] = np.quantile(Sc, q / 100.0, axis=1)
        mp = path("models", f"mos_{side}_{name}.pkl")
        if reuse and mp.exists():
            with open(mp, "rb") as fh:
                model = pickle.load(fh)
        else:
            model = train.train(d[train_m].assign(origin=d.origin[train_m]), tcol, cfg)
        q_all = predict.predict_quantiles(model, d)
        taus = model.taus
        conf = conformal.fit(q_all[val_m], d[tcol].to_numpy()[val_m], taus, groups[val_m], cfg)
        q_cal = conformal.apply(q_all, groups, conf, cfg)
        p_mos = np.zeros_like(p_phys)
        p_mos_nc = np.zeros_like(p_phys)
        ks = (base[:, None] - OFF) + np.arange(2 * OFF + 1)[None, :]
        for i in range(len(d)):
            kk, pp = predict.bin_table(taus, q_cal[i], cfg)
            p_mos[i] = predict.on_bins(predict.to_dict(kk, pp), ks[i])
            kk, pp = predict.bin_table(taus, q_all[i], cfg)
            p_mos_nc[i] = predict.on_bins(predict.to_dict(kk, pp), ks[i])
        p_mos = p_mos / np.maximum(p_mos.sum(1, keepdims=True), 1e-12)
        p_mos_nc = p_mos_nc / np.maximum(p_mos_nc.sum(1, keepdims=True), 1e-12)
        rr = np.clip(realised, 0, 2 * OFF)
        wts = conformal.fit_blend(p_mos[val_m], p_phys[val_m], rr[val_m], groups[val_m], cfg)
        wv = np.array([conformal.blend_weight(wts, g, cfg) for g in groups])
        p_blend = wv[:, None] * p_mos + (1 - wv[:, None]) * p_phys
        rel = conformal.reliability(p_blend[val_m], rr[val_m])
        cov = {f"{lo:.2f}": conformal.coverage(q_cal[val_m], d[tcol].to_numpy()[val_m], taus, lo) for lo in (0.05, 0.1, 0.2)}
        cov_nc = {f"{lo:.2f}": conformal.coverage(q_all[val_m], d[tcol].to_numpy()[val_m], taus, lo) for lo in (0.05, 0.1, 0.2)}
        y = d[tcol].to_numpy()
        rank = (S < y[:, None]).mean(1)
        report[side + "_rank_hist"] = {"2022-2023": np.histogram(rank[train_m], bins=10, range=(0, 1))[0].tolist(),
                                       "2024": np.histogram(rank[val_m], bins=10, range=(0, 1))[0].tolist()}
        rank_c = (Sc < y[:, None]).mean(1)
        report[side + "_rank_hist_calibrated"] = {"2024": np.histogram(rank_c[val_m], bins=10, range=(0, 1))[0].tolist()}
        res[side] = {"raw": p_raw, "physics": p_phys, "mos_noconf": p_mos_nc, "mos": p_mos, "blend": p_blend,
                     "realised": realised, "base": base}
        report[side] = {"spread_scale": sc, "blend_weights": wts, "blend_flags": conformal.blend_flags(wts),
                        "coverage_conformal": cov, "coverage_raw_mos": cov_nc, "reliability": rel,
                        "best_iterations": {str(k): v for k, v in model.best_iterations.items()},
                        "mos_params": getattr(model, "params", None), "mos_selection": getattr(model, "selection", None),
                        "crossed": int((np.diff(q_cal, axis=1) < 0).sum())}
        with open(mp, "wb") as fh:
            pickle.dump(model, fh)
        save_model(model, side, name)
        for t, bst in model.boosters.items():
            bst.save_model(str(path("models", f"mos_{side}_{name}_q{int(round(t * 100)):02d}.lgb")))
        report[side]["conformal"] = conf
    path("params", f"conformal_{name}.json").write_text(json.dumps({s: {"conformal": report[s]["conformal"],
                                                                "blend_weights": report[s]["blend_weights"],
                                                                "spread_scale": report[s]["spread_scale"]}
                                                            for s in ("high", "low")}, indent=1))
    fb = floor_baseline(d[val_m])
    rows = []
    for side in ("high", "low"):
        r = res[side]
        for name in ("raw", "physics", "mos_noconf", "mos", "blend"):
            for period, m in (("2022-2023", train_m), ("2024", val_m)):
                s = score(r[name][m], np.clip(r["realised"][m], -1, 2 * OFF + 1))
                rows.append({"side": side, "model": name, "period": period, **s})
        for name in ("floor", "baseline"):
            s = fb[f"{name}_{side}"]
            rows.append({"side": side, "model": name, "period": "2024", "hit": float(s.mode_hit.mean()),
                         "logloss": float(s.logloss.mean()), "n": int(len(s))})
    sc = pd.DataFrame(rows)
    # stratified 2024 scores
    strat = []
    dv = d[val_m]
    for side in ("high", "low"):
        r = res[side]
        for col in ("is_day", "season", "sector", "sky_class"):
            for v, idx in dv.groupby(col).groups.items():
                ii = np.flatnonzero(val_m)[dv.index.get_indexer(idx)]
                for name in ("raw", "physics", "blend"):
                    s = score(r[name][ii], r["realised"][ii])
                    strat.append({"side": side, "by": col, "value": str(v), "model": name, **s})
    st = pd.DataFrame(strat)
    return sc, st, report, d, res


def frozen_probs(d: pd.DataFrame, samples: dict, name: str) -> dict:
    """Frozen Phase 8 on an assembled frame d and per-side unrounded extreme samples
    ({"high": (n, paths), "low": ...}): spread calibration (2022-2023), MOS (2022-2023), conformal and
    blend weights (2024), loaded from models/mos_<side>_<name>.pkl and params/conformal_<name>.json.
    Nothing is fitted. Probabilities are on bins base-OFF..base+OFF; d gains the q_* physics features."""
    cfg = model_cfg()
    frozen = json.loads(path("params", f"conformal_{name}.json").read_text())
    groups = group_of(d.season, d.is_day)
    res = {}
    for side, real, tcol, qp in (("high", "hi", "h_r", "q_max"), ("low", "lo", "l_r", "q_min")):
        S = samples[side]
        fz = frozen[side]
        base = np.floor(np.median(S, axis=1)).astype(int)
        realised = ((d[real].to_numpy() - (base - OFF)).astype(int) if real in d
                    else np.full(len(d), -1))
        sc = fz["spread_scale"]
        scale = np.array([sc.get(str(g), {"scale": 1.0})["scale"] for g in groups])
        shift = np.array([sc.get(str(g), {"shift": 0.0})["shift"] for g in groups])
        p_raw = sample_probs(S, base, 1.0)
        p_phys = sample_probs(S, base, scale, shift)
        med = np.median(S, axis=1, keepdims=True)
        Sc = med + scale[:, None] * (S - med) + shift[:, None]
        for q in CFG.mos.physics_levels:
            d[f"{qp}_{q:02d}"] = np.quantile(Sc, q / 100.0, axis=1)
        model = load_model(side, name)
        q_all = predict.predict_quantiles(model, d)
        q_cal = conformal.apply(q_all, groups, fz["conformal"], cfg)
        ks = (base[:, None] - OFF) + np.arange(2 * OFF + 1)[None, :]
        p_mos = np.zeros_like(p_phys)
        for i in range(len(d)):
            kk, pp = predict.bin_table(model.taus, q_cal[i], cfg)
            p_mos[i] = predict.on_bins(predict.to_dict(kk, pp), ks[i])
        p_mos = p_mos / np.maximum(p_mos.sum(1, keepdims=True), 1e-12)
        wv = np.array([conformal.blend_weight(fz["blend_weights"], g, cfg) for g in groups])
        p_blend = wv[:, None] * p_mos + (1 - wv[:, None]) * p_phys
        res[side] = {"raw": p_raw, "physics": p_phys, "mos": p_mos, "blend": p_blend, "realised": realised,
                     "base": base, "samples": S, "samples_cal": Sc, "scale": scale, "shift": shift,
                     "y": d[tcol].to_numpy() if tcol in d else np.full(len(d), np.nan),
                     "q_cal": q_cal, "blend_w": wv}
    return res


def apply_frozen(tag: str, first: str, last: str, name: str, use=()) -> tuple[pd.DataFrame, dict]:
    """Frozen Phase 8 on a stored backtest period (see frozen_probs). Nothing is fitted."""
    d, b = assemble(tag, first, last, use)
    ok = d.h_r.notna() & d.l_r.notna()
    keep = ok.to_numpy()
    d = d[ok].reset_index(drop=True)
    res = frozen_probs(d, {"high": b["max_u"][keep], "low": b["min_u"][keep]}, name)
    return d, res
