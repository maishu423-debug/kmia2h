"""Continuous filter + Monte Carlo backtest, one month per worker (SPEC.md Phases 5 and 7).

For a month: spin the EnKF up 6 h before the first origin, run it minute by minute on live
streams (KMIA HFMETAR / METAR; GOES and HRRR when present), snapshot the ensemble at every
30-minute origin and simulate the next 120 minutes through settle(). KMIA 1-minute data is not
read here except by the caller for scoring."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import kmia2h.target as target_ops
from kmia2h.config import CFG, store
from kmia2h.filter import enkf, obs_operators, process_model
from kmia2h.forcing import Inputs, apply_scales, filter_forcing, model_cfg, regime_of, window_forcing
from kmia2h.residual import table as rtable
from kmia2h import simulate
from kmia2h.timeutil import from_ns, to_ns

NUM = ("s_clr", "c", "t_adv", "t_sky", "r", "b_scale", "kappa_scale", "c_scale")


def observations(inp: Inputs, start: pd.Timestamp, extra_obs: pd.DataFrame | None = None) -> list:
    k = inp.kmia[inp.kmia.tmpf.notna()]
    kind = np.where(k.report_type == 1, "hfmetar", np.where(k.tmpc_tenths.notna(), "metar", "hfmetar"))
    d = pd.DataFrame({"ts_utc": k.ts_utc, "available_at": k.available_at, "kind": kind, "value": k.tmpf})
    # one report per valid minute: prefer the tenths-precision METAR
    d["pri"] = (d.kind == "metar").astype(int)
    d = d.sort_values(["ts_utc", "pri"]).drop_duplicates("ts_utc", keep="last").drop(columns="pri")
    if extra_obs is not None and len(extra_obs):
        d = pd.concat([d, extra_obs[["ts_utc", "available_at", "kind", "value"]]], ignore_index=True)
    d = d[(d.ts_utc >= start)]
    return obs_operators.obs_from_frame(d, start)


def regime_arrays(table: dict, ff: pd.DataFrame) -> dict:
    keys = list(zip(ff.season, ff.is_day.astype(str), ff.sky_class, ff.wind))
    cache = {}
    rows = []
    for k in keys:
        if k not in cache:
            cache[k] = rtable.lookup(table, *k)
        rows.append(cache[k])
    return {f: np.array([r[f] for r in rows], float) for f in process_model.REGIME_FIELDS}


def run_month(args) -> str:
    tag, year, month, opts = args
    out = store(f"backtest/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    fo = out / f"{year}{month:02d}.npz"
    if fo.exists() and not opts.get("overwrite"):
        return str(fo)
    t_start = time.time()
    cfg = model_cfg()
    for k, v in opts.get("cfg_override", {}).items():
        sect, key = k.split(".", 1)
        if "." in key:
            a, b = key.split(".")
            cfg[sect][a][b] = v
        else:
            cfg[sect][key] = v
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    spin = start - pd.Timedelta(hours=cfg["filter"]["spinup_hours"])
    extra = opts.get("extra_factory", lambda s, e: {})(spin, end)
    inp = Inputs.load(spin, end, extra=extra)
    minutes = pd.date_range(spin, end, freq="min", tz="UTC", inclusive="left")
    ff = filter_forcing(inp, minutes, cfg)
    ok_forcing = ff[list(NUM)].notna().all(axis=1).to_numpy()
    for c in NUM:
        ff[c] = ff[c].ffill().bfill()
    table = rtable.load(cfg)
    reg = regime_arrays(table, ff)
    step_fn = process_model.make_step({c: ff[c].to_numpy(float) for c in NUM}, reg, cfg)
    rng = np.random.default_rng([cfg["sim"]["seed"], year, month])
    N = int(cfg["enkf"]["members"])
    t0 = float(ff.t_last.dropna().iloc[0]) if ff.t_last.notna().any() else 75.0
    c0 = float(np.nan_to_num(ff.c.iloc[0], nan=0.3))
    x0 = process_model.init_ensemble(N, t0, c0, cfg, rng)
    kf = enkf.EnKF(x0, step_fn, cfg, seed=int(cfg["sim"]["seed"]) + year * 100 + month)
    obs = observations(inp, spin, extra.get("filter_obs"))
    by_avail = defaultdict(list)
    for o in obs:
        if o.available_k >= 0:
            by_avail[max(o.available_k, 0)].append(o)
    step = int(opts.get("step_min", CFG.sim.origin_step_backtest_min))
    origins = pd.date_range(start, end, freq=f"{step}min", inclusive="left")
    ok_of = {int((o - spin).total_seconds() // 60): o for o in origins}
    n_paths = int(opts.get("n_paths", CFG.sim.paths_backtest))
    lead = CFG.sim.lead_min
    # per-minute filtered record (for the residual refit and diagnostics)
    rec_T, rec_Y, rec_sd = np.full(len(minutes), np.nan), np.full(len(minutes), np.nan), np.full(len(minutes), np.nan)
    res = defaultdict(list)
    floor_gap = opts.get("median_gap") or {}
    if 0 in by_avail:
        kf.assimilate(by_avail[0])
    k_end = len(minutes) - 1
    while kf.k < k_end:
        kf.advance()
        if kf.k in by_avail:
            kf.assimilate(by_avail[kf.k])
        x = kf.state
        y = process_model.sensor_temperature(x)
        rec_T[kf.k], rec_Y[kf.k], rec_sd[kf.k] = x[:, 0].mean(), y.mean(), y.std()
        if kf.k in ok_of and opts.get("simulate", True):
            o = ok_of[kf.k]
            try:
                st = inp.state_at(o)
                if not np.isfinite(st["t_last"]) or (o - st["t_ts"]).total_seconds() / 60 > 20:
                    raise ValueError("no recent KMIA observation (ABSTAIN)")
                w = window_forcing(inp, o, cfg, st=st)
                r = regime_of(inp, o, st)
                rp = rtable.sim_params(table, r["season"], r["is_day"], r["sky_class"], r["wind"], cfg)
                snap = kf.snapshot()
                sim = simulate.run(snap, w["forcing"], rp, target_ops, cfg, o, seed=int(cfg["sim"]["seed"]) + kf.k,
                                   n_paths=n_paths, regime=f"{r['season']}|{r['is_day']}|{r['sky_class']}|{r['wind']}",
                                   latest_obs=st["t_last"], cloud=w["cloud"], rain_cells=w["rain_cells"])
            except Exception as e:
                res["fail"].append((str(o), repr(e)[:120]))
                continue
            gk = f"{r['season']}|{r['is_day']}"
            gh, gl = floor_gap.get(gk, (0.0, 0.0))
            fc = sim.forecast
            mean_roll = target_ops.rolling_mean(np.r_[np.full(lead, fc["mean_path"][0]), fc["mean_path"]])[lead:]
            res["origin"].append(int(pd.Timestamp(o).as_unit("ns").value))
            res["max_u"].append(sim.max_unrounded.astype(np.float32))
            res["min_u"].append(sim.min_unrounded.astype(np.float32))
            res["high"].append(sim.high.astype(np.int16))
            res["low"].append(sim.low.astype(np.int16))
            res["mean_max"].append(float(np.max(mean_roll)))
            res["mean_min"].append(float(np.min(mean_roll)))
            res["edge_high"].append(float(target_ops.edge(np.max(mean_roll) + gh)))
            res["edge_low"].append(float(target_ops.edge(np.min(mean_roll) + gl)))
            res["spread"].append(float(sim.checks["spread_of_path_max"]))
            res["steps_ok"].append(bool(sim.checks["steps_within_limit"]))
            res["near_obs"].append(bool(sim.checks["first_minute_near_obs"]))
            res["ens_T_sd"].append(float(snap["ensemble"][:, 0].std()))
            res["ens_Y_mean"].append(float(y.mean()))
            res["t_adv_mean"].append(float(np.mean(w["forcing"]["t_adv"])))
            res["c_mean"].append(float(np.mean(w["forcing"]["c"])))
            res["mean_path"].append(np.asarray(fc["mean_path"], np.float32))
            res["regime"].append(fc["regime"])
            res["sector"].append(r["sector"])
            res["wind_class"].append(w["wind_class"])
            res["cloud_source"].append(w["cloud_source"])
            res["coef"].append(snap["ensemble"][:, 6:10].mean(0).astype(np.float32))
            res["ensemble"].append(snap["ensemble"].astype(np.float32))
    inn = kf.innovations()
    np.savez_compressed(
        fo, origin=np.array(res["origin"], np.int64), max_u=np.array(res["max_u"]), min_u=np.array(res["min_u"]),
        high=np.array(res["high"]), low=np.array(res["low"]),
        **{k: np.array(res[k]) for k in ("mean_max", "mean_min", "edge_high", "edge_low", "spread", "steps_ok",
                                         "near_obs", "ens_T_sd", "ens_Y_mean", "t_adv_mean", "c_mean")},
        mean_path=np.array(res["mean_path"]), regime=np.array(res["regime"], dtype=str),
        sector=np.array(res["sector"], dtype=str), wind_class=np.array(res["wind_class"], dtype=str),
        cloud_source=np.array(res["cloud_source"], dtype=str), coef=np.array(res["coef"]),
        minutes=to_ns(minutes)[len(minutes) - len(pd.date_range(start, end, freq="min", inclusive="left")):],
        rec_T=rec_T[-int((end - start).total_seconds() // 60):].astype(np.float32),
        rec_Y=rec_Y[-int((end - start).total_seconds() // 60):].astype(np.float32),
        rec_sd=rec_sd[-int((end - start).total_seconds() // 60):].astype(np.float32),
        forcing_ok=ok_forcing[-int((end - start).total_seconds() // 60):],
        inn_k=inn.valid_k.to_numpy(), inn_kind=inn.kind.to_numpy().astype(str), inn_val=inn.innovation.to_numpy(),
        inn_var=inn.variance.to_numpy(), fail=np.array([f"{a} {b}" for a, b in res["fail"]], dtype=str),
        seconds=time.time() - t_start)
    if opts.get("save_ensembles"):
        np.savez_compressed(store(f"enkf_snapshots/{tag}_{year}{month:02d}.npz").as_posix(),
                            origin=np.array(res["origin"], np.int64), ensemble=np.array(res["ensemble"]))
    return f"{fo} {time.time() - t_start:.0f}s fails={len(res['fail'])}"


def run(tag: str, first: str, last: str, workers: int = 8, **opts) -> None:
    months = pd.period_range(first, last, freq="M")
    jobs = [(tag, p.year, p.month, opts) for p in months]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_month, jobs):
            print(r, flush=True)


def load(tag: str, first: str, last: str) -> dict:
    months = pd.period_range(first, last, freq="M")
    parts = [np.load(store(f"backtest/{tag}/{p.year}{p.month:02d}.npz"), allow_pickle=False) for p in months]
    keys = [k for k in parts[0].files if not k.startswith(("inn_", "rec_", "minutes", "forcing_ok", "fail", "seconds"))]
    out = {k: np.concatenate([p[k] for p in parts if len(p["origin"])]) for k in keys}
    out["origin"] = from_ns(out["origin"])
    return out
