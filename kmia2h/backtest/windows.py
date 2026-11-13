"""Per-origin window cache: raw forecast forcing (live inputs only, no fitted group ratios:
apply forcing.apply_scales at use) and the 1-minute truth.

data/store/windows/{tag}/{YYYYMM}.npz holds, for every origin of the month on the backtest grid:
forcing arrays (n, 120), live state at origin, regime labels, and the truth (smooth path m,
one-minute t1 with lead minutes, H_r, L_r, settled bins) used only for fitting and scoring."""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from kmia2h.config import CFG, store
from kmia2h.data.qc import clean
from kmia2h.floor.noise_floor import fill_short
from kmia2h.forcing import Inputs, model_cfg, regime_of, window_forcing
from kmia2h.target import round_rule, rolling_mean
from kmia2h.timeutil import from_ns, to_ns

FCOLS = ("s_clr", "c", "t_adv", "t_sky", "r", "b_scale", "kappa_scale", "c_scale")
H = CFG.sim.horizon_min
LEAD = CFG.sim.lead_min


def truth(t1: pd.Series, origins: pd.DatetimeIndex, L: int) -> dict:
    t1f = fill_short(t1, CFG.target.max_interp_gap_min)
    m = t1f.rolling(L, center=True, min_periods=L).mean()
    idx = t1.index.get_indexer(origins)
    rows = idx[:, None] + np.arange(-LEAD + 1, H + 1)[None, :]
    ok = (rows >= 0).all(1) & (rows < len(t1)).all(1)
    rows = np.clip(rows, 0, len(t1) - 1)
    tt = t1f.to_numpy()[rows]
    mm = m.to_numpy()[rows][:, LEAD:]
    raw = t1.to_numpy()[rows][:, LEAD:]
    nmiss = np.isnan(raw).sum(1)
    r = rolling_mean(tt)[:, LEAD:]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hr, lr = np.nanmax(r, 1), np.nanmin(r, 1)
    good = ok & (nmiss <= CFG.target.max_missing_min) & np.isfinite(r).all(1)
    hr, lr = np.where(good, hr, np.nan), np.where(good, lr, np.nan)
    rule = CFG.target.rounding
    return {"t1": tt.astype(np.float32), "m": mm.astype(np.float32), "h_r": hr, "l_r": lr,
            "hi": round_rule(hr, rule), "lo": round_rule(lr, rule), "nmiss": nmiss, "good": good}


def build_month(args) -> str:
    tag, year, month, step = args[:4]
    use = set(args[4]) if len(args) > 4 else set()
    out = store(f"windows/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{year}{month:02d}.npz"
    if f.exists():
        return str(f)
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    origins = pd.date_range(start, end, freq=f"{step}min", inclusive="left")
    cfg = model_cfg()
    from kmia2h import sources
    inp = Inputs.load(start, end, extra=sources.factory(use)(start - pd.Timedelta(hours=8), end))
    k = pd.read_parquet(store("kmia_1min.parquet"), filters=[("ts_utc", ">=", start - pd.Timedelta(hours=1)),
                                                             ("ts_utc", "<", end + pd.Timedelta(hours=3))])
    t1 = clean(k)
    tr = truth(t1, origins, CFG.floor.L_primary)
    F = {c: np.full((len(origins), H), np.nan, np.float32) for c in FCOLS}
    state = {c: np.full(len(origins), np.nan) for c in ("t_last", "dwpf", "sknt", "drct", "t_age_min", "u10", "v10")}
    reg = {c: np.empty(len(origins), dtype=object) for c in ("season", "is_day", "sky_class", "wind", "sector",
                                                             "wind_class", "cloud_source")}
    for i, o in enumerate(origins):
        try:
            st = inp.state_at(o)
            if not np.isfinite(st["t_last"]):
                continue
            w = window_forcing(inp, o, cfg, st=st, scales=False)
        except Exception as e:  # a missing input leaves the row NaN; logged by the caller
            print("window failed", o, repr(e)[:200], file=sys.stderr)
            continue
        for c in FCOLS:
            F[c][i] = w["forcing"][c]
        for c in ("t_last", "dwpf", "sknt", "drct", "u10", "v10"):
            state[c][i] = st[c]
        state["t_age_min"][i] = (o - st["t_ts"]).total_seconds() / 60
        r = regime_of(inp, o, st)
        for c in ("season", "is_day", "sky_class", "wind", "sector"):
            reg[c][i] = r[c]
        reg["wind_class"][i], reg["cloud_source"][i] = w["wind_class"], w["cloud_source"]
    np.savez_compressed(f, origins=to_ns(origins), **{f"f_{c}": v for c, v in F.items()},
                        **{f"s_{c}": v for c, v in state.items()}, **{f"g_{c}": v.astype(str) for c, v in reg.items()},
                        **{f"y_{c}": v for c, v in tr.items()})
    return str(f)


def load(tag: str, first: str, last: str) -> dict:
    months = pd.period_range(first, last, freq="M")
    parts = [np.load(store(f"windows/{tag}/{p.year}{p.month:02d}.npz"), allow_pickle=False) for p in months]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    out["origins"] = from_ns(out["origins"])
    return out


def build(tag: str, first: str, last: str, step: int | None = None, workers: int = 8, use=()) -> None:
    step = step or CFG.sim.origin_step_backtest_min
    months = pd.period_range(first, last, freq="M")
    jobs = [(tag, p.year, p.month, step, tuple(use)) for p in months]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(build_month, jobs):
            print(r, flush=True)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3], use=sys.argv[4].split(",") if len(sys.argv) > 4 else ())
