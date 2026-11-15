"""The live run (SPEC.md Phase 10): one pass of the frozen pipeline at an origin. Nothing is refit.

Modes
  replay   historical origin, bit-for-bit the Phase 9 backtest: the filter runs from the backtest's
           month spin-up (month start - 6 h) with the same seeds and 2,000 paths, stopping at the origin.
  spinup   historical origin, as a live run would see it: 6 h spin-up before the origin, 4,000 paths,
           graceful degradation on. Used by the outage tests.
  live     --origin now: fetch every live source, restart the filter from the persisted state when it
           is < 6 h old (else spin up), degradation on, persist the new state.

Graceful degradation (spinup / live; SPEC.md Phase 10):
  GOES stale  -> use the last scan up to 3 h old, the filter carries C_eff, cloud perturbation x2
  MRMS stale  -> R = 0 (no cells) and the jump rate raised to the 90th percentile of the season x day/night regimes
  HRRR late   -> the previous cycle at a longer lead (the sources layer takes the newest cycle within 6 h)
  KMIA stale  -> no HFMETAR / METAR in the last 20 min: ABSTAIN. A neighbour is never substituted.
Replay keeps the backtest's semantics exactly and only reports what is degraded."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kmia2h.config import CFG, ROOT, path, store
from kmia2h.live import compose

FZ = CFG.frozen
USE = tuple(FZ.sources)
OUTAGE_KINDS = ("goes", "mrms", "hrrr", "kmia")
GOES_FALLBACK_AGE = pd.Timedelta(hours=3)
STATE_MAX_AGE = pd.Timedelta(hours=CFG.filter.spinup_hours)
# The state is persisted this long before the origin, so the next run re-assimilates reports that were
# valid after it but reached IEM late (IEM HFMETAR runs ~30 min behind real time).
STATE_LAG = pd.Timedelta(minutes=60)
LIVE_DIR = ROOT / CFG.paths.out / "live"
# KMIA2H_STATE_FILE: where the live filter state is persisted (the website keeps it on its /data volume)
STATE_FILE = Path(os.environ.get("KMIA2H_STATE_FILE") or ROOT / "data" / "live" / "filter_state.npz")


# ---- source tables ---------------------------------------------------------------------------

def _clip(name: str, col: str, lo, hi) -> pd.DataFrame | None:
    p = store(name)
    if not p.exists():
        return None
    d = pd.read_parquet(p, filters=[(col, ">=", lo), (col, "<=", hi)])
    for c in d.columns:
        if isinstance(d[c].dtype, pd.DatetimeTZDtype):
            d[c] = d[c].astype("datetime64[ns, UTC]")
    return d.reset_index(drop=True)


def store_tables(lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    """Store tables clipped to [lo, hi] (what a live fetch would have returned, and more)."""
    return {"obs": _clip("obs.parquet", "ts_utc", lo, hi), "ndbc": _clip("ndbc.parquet", "ts_utc", lo, hi),
            "goes_feat": _clip("goes_feat.parquet", "ts_utc", lo, hi),
            "mrms_feat": _clip("mrms_feat.parquet", "ts_utc", lo, hi),
            "nwp_point": _clip("nwp_point.parquet", "cycle_utc", lo - pd.Timedelta(hours=6), hi)}


def apply_outage(tables: dict, kinds, origin: pd.Timestamp, minutes: int = 60) -> dict:
    """Remove the last `minutes` of a source before the origin (an outage that began then)."""
    t = dict(tables)
    cut = origin - pd.Timedelta(minutes=minutes)
    for k in kinds:
        if k not in OUTAGE_KINDS:
            raise ValueError(f"unknown outage {k!r}; choose from {OUTAGE_KINDS}")
    if "goes" in kinds and t.get("goes_feat") is not None:
        t["goes_feat"] = t["goes_feat"][t["goes_feat"].ts_utc <= cut]
    if "mrms" in kinds and t.get("mrms_feat") is not None:
        t["mrms_feat"] = t["mrms_feat"][t["mrms_feat"].ts_utc <= cut]
    if "hrrr" in kinds and t.get("nwp_point") is not None:
        n = t["nwp_point"]
        t["nwp_point"] = n[n.cycle_utc + pd.Timedelta(minutes=CFG.latency_min.hrrr) <= cut]
    if "kmia" in kinds:
        o = t["obs"]
        t["obs"] = o[(o.station != CFG.station) | (o.ts_utc <= cut)]
    return t


def _age(d: pd.DataFrame | None, origin: pd.Timestamp, tcol="ts_utc", acol="available_at"):
    if d is None or d.empty or acol not in d:
        return None
    a = d[d[acol] <= origin]
    if "cf_disc" in a:
        a = a[a.cf_disc.notna()]
    return None if a.empty else origin - a[tcol].max()


def degradation(tables: dict, origin: pd.Timestamp) -> dict:
    """What is missing or stale at the origin, and the fallbacks it calls for."""
    from kmia2h.sources import MAX_GOES_AGE, MAX_MRMS_AGE
    out_ = {"notes": [], "goes_max_age": None, "cloud_sigma_x": 1.0, "mrms_missing": False, "kmia_age_min": None}
    g = _age(tables.get("goes_feat"), origin)
    if "goes" in USE and (g is None or g > MAX_GOES_AGE):
        out_["cloud_sigma_x"] = 2.0
        if g is not None and g <= GOES_FALLBACK_AGE:
            out_["goes_max_age"] = GOES_FALLBACK_AGE
            out_["notes"].append(f"goes: last scan {g.total_seconds() / 60:.0f} min old, used; cloud perturbation x2")
        else:
            out_["notes"].append("goes: no scan within 3 h; cloud from HRRR / METAR sky; cloud perturbation x2")
    m = _age(tables.get("mrms_feat"), origin)
    if "mrms" in USE and (m is None or m > MAX_MRMS_AGE):
        out_["mrms_missing"] = True
        out_["notes"].append("mrms: missing; R = 0 and jump rate at the regime 90th percentile")
    n = tables.get("nwp_point")
    if "hrrr" in USE:
        newest = (origin - pd.Timedelta(minutes=CFG.latency_min.hrrr)).floor("h")
        avail = None if n is None or n.empty else n[n.cycle_utc + pd.Timedelta(minutes=CFG.latency_min.hrrr) <= origin]
        c = None if avail is None or avail.empty else avail.cycle_utc.max()
        if c is None or origin - c > pd.Timedelta(hours=6):
            out_["notes"].append("hrrr: no cycle within 6 h; HRRR terms off")
        elif c < newest:
            out_["notes"].append(f"hrrr: cycle {c:%H}Z used (newest expected {newest:%H}Z), lead +{(newest - c).total_seconds() / 3600:.0f} h")
    o = tables.get("obs")
    k = None if o is None else o[(o.station == CFG.station) & o.tmpf.notna() & (o.available_at <= origin)]
    if k is None or k.empty:
        out_["notes"].append("kmia: no reports")
    else:
        out_["kmia_age_min"] = float((origin - k.ts_utc.max()).total_seconds() / 60)
        if out_["kmia_age_min"] > 20:
            out_["notes"].append(f"kmia: newest report {out_['kmia_age_min']:.0f} min old -> ABSTAIN")
    return out_


def mrms_hook(table: dict):
    """Jump rate raised to the 90th percentile of the season x day/night regimes (MRMS missing)."""
    def hook(rp, r):
        pre = f"{r['season']}|{r['is_day']}|"
        lam = [v["lam_per_hour"] for k, v in table.items()
               if isinstance(v, dict) and k.startswith(pre) and "lam_per_hour" in v]
        if lam:
            rp = dict(rp)
            rp["lam_per_hour"] = max(float(rp.get("lam_per_hour", 0.0)), float(np.quantile(lam, 0.9)))
        return rp
    return hook


# ---- filter state ----------------------------------------------------------------------------

def load_state(origin: pd.Timestamp):
    """(time, ensemble) of the persisted filter state if it is 0-6 h before the origin, else None."""
    if not STATE_FILE.exists():
        return None
    s = np.load(STATE_FILE)
    t = pd.Timestamp(int(s["t_ns"]), tz="UTC")
    if not (pd.Timedelta(0) < origin - t <= STATE_MAX_AGE):
        return None
    return t, s["ensemble"].astype(float)


def save_state(t: pd.Timestamp, ensemble: np.ndarray) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(STATE_FILE, t_ns=np.int64(t.value), ensemble=ensemble.astype(np.float64))


# ---- the run ---------------------------------------------------------------------------------

@dataclass
class Run:
    origin: pd.Timestamp
    mode: str
    window_start: pd.Timestamp | None = None
    outage: tuple = ()
    outage_min: int = 60
    seed: int | None = None
    n_paths: int | None = None
    tables: dict | None = None            # live: fetched tables
    fetch_log: dict = field(default_factory=dict)
    use_state: bool = True


def _bins(ks, p, top: int | None = None) -> dict:
    keep = p >= 1e-4
    return {str(int(k)): round(float(v), 5) for k, v in zip(np.asarray(ks)[keep], np.asarray(p)[keep])}


def forecast(run: Run) -> dict:
    from kmia2h.backtest import backtest
    from kmia2h.features import Features
    from kmia2h.mos import pipeline
    from kmia2h.phases import phase9
    t_all = time.time()
    o = pd.Timestamp(run.origin).floor("min")
    spin_h = pd.Timedelta(hours=CFG.filter.spinup_hours)
    timing = {}
    opts = {"sources": USE, "simulate": True}
    if run.seed is not None:
        opts["cfg_override"] = {"sim.seed": int(run.seed)}

    # Source tables around the origin: features, MOS source features, degradation checks.
    if run.tables is not None:
        tables = run.tables
    else:
        tables = store_tables(o - pd.Timedelta(hours=12), o + pd.Timedelta(hours=3))
    if run.outage:
        tables = apply_outage(tables, run.outage, o, run.outage_min)
    deg = degradation(tables, o)

    if run.mode == "replay":
        start = o.floor("D").replace(day=1)
        end = start + pd.offsets.MonthBegin(1)
        spin, seed_key, x0 = start - spin_h, (start.year, start.month), None
        opts["n_paths"] = int(run.n_paths or CFG.sim.paths_backtest)
        span_tables = tables_for_span(spin, end, run.outage, o, run.outage_min) if run.outage else None
        state_note = "replay from the backtest month spin-up"
    else:
        st = load_state(o) if (run.mode == "live" and run.use_state) else None
        if st is not None:
            spin, x0 = st
            state_note = f"restarted from persisted state at {spin:%Y-%m-%dT%H:%MZ}"
        else:
            spin, x0 = o - spin_h, None
            state_note = "6 h spin-up"
        end = o + pd.Timedelta(minutes=1)
        seed_key = (o.year, o.month)
        opts["n_paths"] = int(run.n_paths or CFG.sim.paths_live)
        span_tables = tables
        if deg["goes_max_age"] is not None:
            opts["goes_max_age"] = deg["goes_max_age"]
        if deg["cloud_sigma_x"] != 1.0:
            from kmia2h.forcing import model_cfg
            opts.setdefault("cfg_override", {})["simulate.cloud_sigma"] = (
                float(model_cfg()["simulate"]["cloud_sigma"]) * deg["cloud_sigma_x"])
    cfg = backtest.setup_cfg(opts)
    if run.mode != "replay" and deg["mrms_missing"]:
        from kmia2h.residual import table as rtable
        opts["rp_hook"] = mrms_hook(rtable.load(cfg))

    t = time.time()
    kmia_obs = tables.get("obs")
    if kmia_obs is None or kmia_obs[kmia_obs.station == CFG.station].empty:
        return _abstain(run, o, "no KMIA reports", deg, timing, state_note, t_all)
    t_state = o - STATE_LAG
    if run.mode == "live" and t_state > spin:
        opts["state_k"] = int((t_state - spin).total_seconds() // 60)
    span = backtest.run_span(spin, end, [o], opts, seed_key, tables=span_tables, x0=x0, cfg=cfg,
                             stop_at_last=True, keep=True)
    timing["filter_and_simulation_s"] = round(time.time() - t, 2)
    res = span["res"]
    if run.mode == "live" and span.get("state_x") is not None:
        save_state(t_state, span["state_x"])
    if not res.get("origin"):
        reason = res["fail"][0][1] if res.get("fail") else "no forecast"
        if "ABSTAIN" in reason:
            reason = f"no KMIA HFMETAR / METAR in the last 20 min (newest {deg['kmia_age_min']:.0f} min old)"
        return _abstain(run, o, reason, deg, timing, state_note, t_all)
    one = res["kept"][0]

    # Frozen MOS, conformal and blend.
    t = time.time()
    b = {k: np.array(v) for k, v in res.items() if k not in ("kept", "fail", "ensemble", "regime", "sector",
                                                               "wind_class", "cloud_source", "mean_path", "coef")}
    b["origin"] = pd.DatetimeIndex([o])
    feats = Features(obs=tables["obs"], ndbc=tables.get("ndbc")).features_at([o])
    d = pipeline.frame_from(b, feats, USE, tables=tables)
    pr = pipeline.frozen_probs(d, {"high": b["max_u"], "low": b["min_u"]}, FZ.mos_name)
    timing["mos_s"] = round(time.time() - t, 2)

    rule = json.loads(path("params", "abstain.json").read_text())
    r = one["r"]
    out_ = {"origin_utc": f"{o:%Y-%m-%dT%H:%MZ}", "run_utc": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ}",
            "mode": run.mode, "status": "OK", "window_utc": [f"{o + pd.Timedelta(minutes=1):%H:%MZ}",
                                                         f"{o + pd.Timedelta(minutes=CFG.sim.horizon_min):%H:%MZ}"],
            "regime": f"{'day' if r['is_day'] else 'night'}-{r['sector']}-{r['sky_class']}",
            "regime_key": one["sim"].forecast["regime"], "latest_kmia_f": one["st"]["t_last"],
            "latest_kmia_utc": f"{one['st']['t_ts']:%Y-%m-%dT%H:%MZ}"}
    acts = {}
    for side in ("high", "low"):
        rr = pr[side]
        ks = rr["base"][0] - pipeline.OFF + np.arange(2 * pipeline.OFF + 1)
        p = rr["blend"][0]
        g = pd.DataFrame({"edge": [float(d[f"edge_{side}"].iloc[0])], "blend_pmode": [float(p.max())],
                          "spread": [float(rr["samples_cal"][0].std())], "is_day": [bool(r["is_day"])],
                          "season": [r["season"]], "sky_class": [r["sky_class"]], "sector": [r["sector"]]})
        pc = float(phase9.abstain_p(rule, side, g)[0])
        act = pc >= rule[side]["threshold"]
        acts[side] = act
        out_[side] = {"mode": int(ks[p.argmax()]), "p_mode": round(float(p.max()), 4), "bins": _bins(ks, p),
                      "edge": round(float(d[f"edge_{side}"].iloc[0]), 3), "p_correct": round(pc, 4),
                      "threshold": rule[side]["threshold"], "action": "ACT" if act else "ABSTAIN",
                      "physics_bins": _bins(ks, rr["physics"][0]), "blend_weight": float(rr["blend_w"][0])}
    out_["action"] = ("ACT" if all(acts.values()) else "ABSTAIN" if not any(acts.values())
                      else " / ".join(f"{'ACT' if a else 'ABSTAIN'} {s}" for s, a in acts.items()))
    if run.window_start is not None:
        out_["day"] = compose_day(run, o, tables, pr, one, d)
    out_.update({"degraded": deg["notes"], "filter": state_note, "sources_fetch": run.fetch_log,
                 "config": {"backtest_tag": FZ.backtest_tag, "mos_name": FZ.mos_name, "sources": list(USE),
                            "seed": int(cfg["sim"]["seed"]), "n_paths": opts["n_paths"],
                            "outage": list(run.outage)}})
    timing["total_s"] = round(time.time() - t_all, 2)
    out_["timing"] = timing
    out_["summary"] = summary(out_)
    out_["_arrays"] = {"max_u": b["max_u"][0], "min_u": b["min_u"][0], "p_high": pr["high"]["blend"][0],
                       "p_low": pr["low"]["blend"][0]}
    return out_


def tables_for_span(spin, end, outage, o, minutes) -> dict:
    t = store_tables(spin - pd.Timedelta(hours=8), end + pd.Timedelta(hours=3))
    return apply_outage(t, outage, o, minutes)


def compose_day(run: Run, o: pd.Timestamp, tables: dict, pr: dict, one: dict, d: pd.DataFrame) -> dict:
    """Observed part [window_start, origin] + forecast part -> day high / low bins (SPEC.md step 6)."""
    from kmia2h.mos import pipeline
    from kmia2h.target import rolling_mean
    ws = pd.Timestamp(run.window_start)
    day_end = ws + pd.Timedelta(days=1)
    if not (ws <= o < day_end):
        raise ValueError(f"origin {o} is outside the climate day starting {ws}")
    pm = compose.load()
    el_h = (o - ws).total_seconds() / 3600
    k = tables["obs"]
    k = k[k.station == CFG.station]
    hx, lx, n_rep = compose.report_extremes(k, ws, o)
    m_left = int((day_end - o).total_seconds() // 60) - 1        # forecast minutes o+1 .. day_end-1
    H = CFG.sim.horizon_min
    out_ = {"window_start_utc": f"{ws:%Y-%m-%dT%H:%MZ}", "day_end_utc": f"{day_end:%Y-%m-%dT%H:%MZ}",
            "elapsed_h": round(el_h, 2), "reports_in_elapsed": n_rep,
            "report_max_f": hx, "report_min_f": lx}
    covers = o + pd.Timedelta(minutes=min(H, m_left))
    out_["covers_to_utc"] = f"{covers:%Y-%m-%dT%H:%MZ}"
    out_["complete"] = bool(m_left <= H)
    if not out_["complete"]:
        out_["note"] = ("distribution of the extreme over window start .. origin + 2 h; the rest of the day is not "
                        "forecast, so the day high can only be higher and the day low only lower")
    lead = CFG.sim.lead_min
    if m_left < H:   # the 2-hour window crosses LST midnight: truncated physics paths, spread-calibrated
        paths = one["sim"].paths[:, : lead + m_left].astype(float)
        roll = rolling_mean(paths)[:, lead:]
        trunc = {"high": roll.max(1), "low": roll.min(1)}
        out_["forecast_part"] = "physics, truncated at LST midnight (MOS is trained on full 2-hour windows)"
    else:
        trunc = None
        out_["forecast_part"] = "final model (blend)"
    for side in ("high", "low"):
        rr = pr[side]
        if trunc is None:
            ks_fc = rr["base"][0] - pipeline.OFF + np.arange(2 * pipeline.OFF + 1)
            p_fc = rr["blend"][0]
        else:
            S = trunc[side][None, :]
            base = np.floor(np.median(S, axis=1)).astype(int)
            ks_fc = base[0] - pipeline.OFF + np.arange(2 * pipeline.OFF + 1)
            p_fc = pipeline.sample_probs(S, base, rr["scale"][:1], rr["shift"][:1])[0]
        rep = hx if side == "high" else lx
        if n_rep == 0 or not np.isfinite(rep):
            ks, p = ks_fc, p_fc
            out_.setdefault("warnings", []).append("no KMIA reports in the elapsed part: forecast part only")
        elif m_left < 1:
            ks, p = compose.observed_pmf(pm, side, rep, el_h)
        else:
            ks_o, p_o = compose.observed_pmf(pm, side, rep, el_h)
            ks, p = compose.combine(ks_o, p_o, ks_fc, p_fc, side)
        out_[side] = {"mode": int(ks[np.argmax(p)]), "p_mode": round(float(p.max()), 4), "bins": _bins(ks, p)}
    return out_


def _abstain(run: Run, o, reason, deg, timing, state_note, t_all) -> dict:
    timing["total_s"] = round(time.time() - t_all, 2)
    out_ = {"origin_utc": f"{o:%Y-%m-%dT%H:%MZ}", "run_utc": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ}",
            "mode": run.mode, "status": "ABSTAIN", "reason": reason, "action": "ABSTAIN",
            "degraded": deg["notes"], "filter": state_note, "sources_fetch": run.fetch_log,
            "config": {"backtest_tag": FZ.backtest_tag, "mos_name": FZ.mos_name, "outage": list(run.outage)},
            "timing": timing}
    out_["summary"] = f"ABSTAIN | {reason}"
    return out_


def summary(f: dict) -> str:
    s = (f"high: {f['high']['mode']} ({f['high']['p_mode']:.2f}) | low: {f['low']['mode']} ({f['low']['p_mode']:.2f}) | "
         f"regime {f['regime']} | edge_high {f['high']['edge']:.2f} | {f['action']}")
    if "day" in f:
        dd = f["day"]
        s += (f"\nday {dd['window_start_utc'][:10]} LST: high {dd['high']['mode']} ({dd['high']['p_mode']:.2f}) | "
              f"low {dd['low']['mode']} ({dd['low']['p_mode']:.2f}) | to {dd['covers_to_utc'][11:]}"
              f"{'' if dd['complete'] else ' (partial day)'}")
    if f.get("degraded"):
        s += "\ndegraded: " + "; ".join(f["degraded"])
    return s


def write(f: dict, out_path=None) -> None:
    out_path = out_path or LIVE_DIR / "forecast.json"
    out_path = path_obj(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in f.items() if not k.startswith("_")}
    out_path.write_text(json.dumps(clean, indent=1, default=_json))
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_DIR / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(clean, default=_json) + "\n")


def path_obj(p):
    from pathlib import Path
    return Path(p)


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return str(x)
