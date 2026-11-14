"""Phase 9: score the frozen pipeline on a period it never saw (SPEC.md Phase 9).

Everything here is evaluation. The models, spread calibration, conformal offsets, blend weights and
the abstain rule are loaded; the only fit is the abstain rule, and it uses 2024 alone.

Per origin and side the table holds the realised settled value, the unrounded realised extreme, the
log-loss / mode hit / CRPS of each model (floor, baseline, raw ensemble, calibrated physics, MOS,
final blend), sharpness, edge, regime and the raw-ensemble rank of the realised extreme.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kmia2h.config import CFG, out, path, store
from kmia2h.floor import noise_floor
from kmia2h.mos import conformal, pipeline

SIDES = ("high", "low")
MODELS = ("raw", "physics", "mos", "blend")
REF = ("floor", "baseline")
P_FLOOR = 1e-4
EDGE_BINS = [0.0, 0.1, 0.2, 0.3, 0.5]
OFF = pipeline.OFF


# --- per-origin scores ------------------------------------------------------------------------

def crps_discrete(p: np.ndarray, ks: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of a distribution on whole-degree points ks (rows of p) against the continuous outcome y:
    E|X - y| - E|X - X'| / 2."""
    a = (p * np.abs(ks - y[:, None])).sum(1)
    diff = np.abs(ks[:, :, None] - ks[:, None, :])
    b = np.einsum("ni,nij,nj->n", p, diff, p)
    return a - 0.5 * b


def per_origin(p: np.ndarray, realised: np.ndarray, ks: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    ok = (realised >= 0) & (realised < p.shape[1])
    pr = np.where(ok, p[np.arange(len(p)), np.clip(realised, 0, p.shape[1] - 1)], 0.0)
    return pd.DataFrame({"logloss": -np.log(np.maximum(pr, P_FLOOR)), "hit": (p.argmax(1) == realised).astype(float),
                         "crps": crps_discrete(p, ks, y), "sharp": p.max(1), "pmode": p.max(1)})


def floor_baseline(origins: pd.DatetimeIndex) -> dict:
    """Phase 3 oracle and persistence+trend on these origins, residual pools from 2016-2023."""
    ft = pd.read_parquet(store("floor_table_L21_full.parquet"))
    fit0, fit1 = CFG.periods.floor_fit
    train = pd.Series((ft.index >= pd.Timestamp(fit0, tz="UTC")) &
                      (ft.index <= pd.Timestamp(fit1, tz="UTC") + pd.Timedelta(hours=23, minutes=59)), index=ft.index)
    test = pd.Series(ft.index.isin(origins), index=ft.index)
    out_ = {}
    for name, pc, rc, real, cont in (("floor_high", "Hm", "eH", "hi", "Hr"), ("floor_low", "Lm", "eL", "lo", "Lr"),
                                     ("baseline_high", "bH", "ebH", "hi", "Hr"), ("baseline_low", "bL", "ebL", "lo", "Lr")):
        ok = ft[pc].notna()
        d, key, tr, te = ft[ok], ft.regime[ok], train[ok], test[ok]
        rows = []
        for g, sub in d[te].groupby(key[te]):
            res = np.sort(d.loc[tr & (key == g), rc].values)
            ks, p = noise_floor.bin_probs(sub[pc].values, res)
            realised = (sub[real].values - ks[:, 0]).astype(int)
            s = per_origin(p, realised, ks.astype(float), sub[cont].values)
            s.index = sub.index
            rows.append(s)
        out_[name] = pd.concat(rows).reindex(origins)
    out_["rain_win"] = ft.rain_win.reindex(origins)
    return out_


def table(tag: str, first: str, last: str, name: str, use) -> pd.DataFrame:
    """One row per (origin, side) with every model's per-origin scores."""
    d, res = pipeline.apply_frozen(tag, first, last, name, use)
    origins = pd.DatetimeIndex(d.origin)
    fb = floor_baseline(origins)
    parts = []
    for side in SIDES:
        r = res[side]
        ks = (r["base"][:, None] - OFF) + np.arange(2 * OFF + 1)[None, :]
        t = pd.DataFrame({"origin": origins, "side": side, "y": r["y"], "realised_k": r["realised"] + r["base"] - OFF,
                          "season": d.season.to_numpy(), "is_day": d.is_day.astype(str).to_numpy() == "True",
                          "sector": d.sector.to_numpy(), "sky_class": d.sky_class.to_numpy(),
                          "hour_lst": d.hour_lst.to_numpy(), "rain_win": fb["rain_win"].to_numpy(),
                          "edge": d[f"edge_{side}"].to_numpy(),
                          "spread": r["samples_cal"].std(1),
                          "rank": (r["samples"] < r["y"][:, None]).mean(1)})
        for m in MODELS:
            s = per_origin(r[m], r["realised"], ks.astype(float), r["y"])
            for c in s.columns:
                t[f"{m}_{c}"] = s[c].to_numpy()
        for ref in REF:
            s = fb[f"{ref}_{side}"]
            for c in ("logloss", "hit", "crps"):
                t[f"{ref}_{c}"] = s[c].to_numpy()
        t["blend_mode_k"] = r["blend"].argmax(1) + r["base"] - OFF
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t["day"] = (t.origin - pd.Timedelta(hours=5)).dt.date  # LST day, the block for the bootstrap
    t["edge_bin"] = pd.cut(t.edge, EDGE_BINS, include_lowest=True).astype(str)
    return t, res, d


# --- aggregation ------------------------------------------------------------------------------

METRICS = ("logloss", "hit", "crps")


def boot_se(t: pd.DataFrame, col: str, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """Mean of col and its block-bootstrap standard error (blocks = LST days)."""
    v = t[[col, "day"]].dropna()
    if v.empty:
        return np.nan, np.nan
    g = v.groupby("day")[col].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(means.std(ddof=1))


def summarise(t: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    rows = []
    groups = [("all", t)] if by is None else list(t.groupby(by, observed=True))
    for gv, g in groups:
        for side, gs in g.groupby("side"):
            for m in REF + MODELS:
                row = {"by": by or "all", "value": str(gv), "side": side, "model": m, "n": int(gs[f"{m}_logloss"].notna().sum())}
                for c in METRICS:
                    mean, se = boot_se(gs, f"{m}_{c}")
                    row[c], row[f"{c}_se"] = mean, se
                if m in MODELS:
                    row["sharpness"] = float(gs[f"{m}_sharp"].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def reliability(t: pd.DataFrame, res: dict, model: str = "blend") -> pd.DataFrame:
    rows = []
    for side in SIDES:
        r = res[side]
        rr = np.clip(r["realised"], 0, 2 * OFF)
        for b in conformal.reliability(r[model], rr):
            rows.append({"side": side, **b})
    return pd.DataFrame(rows)


def rank_hist(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, g in t.groupby("side"):
        h = np.histogram(g["rank"], bins=10, range=(0, 1))[0]
        rows.append({"side": side, **{f"r{i}": int(v) for i, v in enumerate(h)}})
    return pd.DataFrame(rows)


# --- abstain rule -----------------------------------------------------------------------------

ABSTAIN_FEATURES = ("edge", "pmode", "spread", "is_day", "wet", "sky_clear", "sky_sct", "sky_bkn", "sky_ovc",
                    "sec_onshore", "sec_offshore")


def abstain_x(t: pd.DataFrame) -> np.ndarray:
    x = pd.DataFrame({"edge": t.edge, "pmode": t.blend_pmode, "spread": t.spread, "is_day": t.is_day.astype(float),
                      "wet": (t.season == "wet").astype(float)})
    for s in ("clear", "sct", "bkn", "ovc"):
        x[f"sky_{s}"] = (t.sky_class == s).astype(float)
    for s in ("onshore", "offshore"):
        x[f"sec_{s}"] = (t.sector == s).astype(float)
    return x[list(ABSTAIN_FEATURES)].fillna(0.0).to_numpy(float)


def fit_abstain(t24: pd.DataFrame, target_hit: float = 0.85, min_share: float = 0.01) -> dict:
    """Per side: logistic P(mode correct | edge, mode probability, regime, spread) on 2024, then the
    lowest threshold whose acted-on 2024 origins hit >= target_hit with at least min_share acted on;
    if no threshold reaches it, the threshold with the best hit rate at >= min_share (logged)."""
    from sklearn.linear_model import LogisticRegression
    rule = {"features": list(ABSTAIN_FEATURES), "target_hit": target_hit, "min_share": min_share, "fit_period": "2024"}
    for side, g in t24.groupby("side"):
        x, y = abstain_x(g), g.blend_hit.to_numpy()
        mu, sd = x.mean(0), x.std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=2000).fit((x - mu) / sd, y)
        p = lr.predict_proba((x - mu) / sd)[:, 1]
        order = np.sort(np.unique(np.round(p, 4)))
        best, chosen = None, None
        for th in order:
            act = p >= th
            if act.mean() < min_share:
                break
            hit = y[act].mean()
            if best is None or hit > best[1]:
                best = (float(th), float(hit), float(act.mean()))
            if chosen is None and hit >= target_hit:
                chosen = (float(th), float(hit), float(act.mean()))
        reached = chosen is not None
        th, hit, share = chosen if reached else best
        rule[side] = {"mean": mu.tolist(), "sd": sd.tolist(), "coef": lr.coef_[0].tolist(),
                      "intercept": float(lr.intercept_[0]), "threshold": th, "hit_2024": hit, "share_2024": share,
                      "target_reached": reached}
    return rule


def abstain_p(rule: dict, side: str, g: pd.DataFrame) -> np.ndarray:
    r = rule[side]
    z = (abstain_x(g) - np.array(r["mean"])) / np.array(r["sd"])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(r["coef"]) + r["intercept"])))


def evaluate_abstain(rule: dict, t: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for side, g in t.groupby("side"):
        act = abstain_p(rule, side, g) >= rule[side]["threshold"]
        a = g[act]
        rng = np.random.default_rng(seed)
        days = g.day.unique()
        hits_by_day = a.groupby("day").blend_hit.agg(["sum", "size"]).reindex(days).fillna(0)
        s, n = hits_by_day["sum"].to_numpy(), hits_by_day["size"].to_numpy()
        idx = rng.integers(0, len(days), (n_boot, len(days)))
        with np.errstate(invalid="ignore"):
            bs = s[idx].sum(1) / n[idx].sum(1)
        bs = bs[np.isfinite(bs)]
        rows.append({"side": side, "n": len(g), "acted": int(act.sum()), "share_acted": float(act.mean()),
                     "hit_acted": float(a.blend_hit.mean()) if len(a) else np.nan,
                     "ci_lo": float(np.quantile(bs, 0.025)) if len(bs) else np.nan,
                     "ci_hi": float(np.quantile(bs, 0.975)) if len(bs) else np.nan,
                     "hit_all": float(g.blend_hit.mean()), "threshold": rule[side]["threshold"]})
    return pd.DataFrame(rows)


# --- stability (Test A vs Test B) -------------------------------------------------------------

def stability(sa: pd.DataFrame, sb: pd.DataFrame) -> pd.DataFrame:
    key = ["by", "value", "side", "model"]
    m = sa.merge(sb, on=key, suffixes=("_A", "_B"))
    rows = []
    for _, r in m.iterrows():
        for c in METRICS:
            diff = r[f"{c}_B"] - r[f"{c}_A"]
            se = float(np.hypot(r[f"{c}_se_A"], r[f"{c}_se_B"]))
            rows.append({**{k: r[k] for k in key}, "metric": c, "A": r[f"{c}_A"], "B": r[f"{c}_B"], "diff": diff,
                         "se_diff": se, "n_A": r["n_A"], "n_B": r["n_B"],
                         "flag": bool(np.isfinite(se) and se > 0 and abs(diff) > 2 * se)})
    return pd.DataFrame(rows)


# --- attribution of the worst windows ---------------------------------------------------------

FORCING_NAMES = ("s_clr", "c", "t_adv", "r", "b_scale", "kappa_scale")


def worst_windows(t: pd.DataFrame, tag: str, n: int = 50, use=()) -> pd.DataFrame:
    """The n origins with the largest summed high+low blended log-loss: a PNG each and a failure class.

    Classes, first match wins:
      state      - the ensemble mean at the origin is > 1.5 F from the 1-minute truth (filter / obs problem)
      rain       - rain fell in the window, or rain cells were forecast and none fell
      cloud      - daytime, and forecast cloud at window end differs from GOES there by > 0.4
      sea_breeze - daytime and the sea-breeze switch fired, or a wet-season daytime high bust (> 2 F too warm)
                   with no switch
      coefficients - everything else: the forcing looked right and the ODE/residual still missed
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from kmia2h.backtest import windows

    w = t.pivot_table(index="origin", columns="side", values="blend_logloss")
    worst = w.sum(1).sort_values(ascending=False).head(n).index
    od = out(9, "worst")
    od.mkdir(parents=True, exist_ok=True)
    ends = pd.DatetimeIndex(worst) + pd.Timedelta(minutes=CFG.sim.horizon_min)
    cf_end = pipeline.source_features(ends.sort_values(), ("goes",))["cf_disc"].reindex(ends).to_numpy()
    rows = []
    for i, o in enumerate(worst):
        o = pd.Timestamp(o)
        ym = f"{o.year}{o.month:02d}"
        b = np.load(store(f"backtest/{tag}/{ym}.npz"))
        j = int(np.flatnonzero(b["origin"] == o.as_unit("ns").value)[0])
        wd = np.load(store(f"windows/v1/{ym}.npz"))
        jw = int(np.flatnonzero(wd["origins"] == o.as_unit("ns").value)[0])
        t1 = wd["y_t1"][jw]
        lead = len(t1) - CFG.sim.horizon_min
        mp = b["mean_path"][j]
        sp = b["sample_paths"][j] if "sample_paths" in b.files else None
        F = b["forcing_terms"][j] if "forcing_terms" in b.files else None
        ncell = int(b["n_rain_cells"][j]) if "n_rain_cells" in b.files else 0
        row = t[(t.origin == o)].set_index("side")
        hi, lo = row.loc["high"], row.loc["low"]
        state_err = float(b["ens_Y_mean"][j] - t1[lead - 1]) if np.isfinite(t1[lead - 1]) else np.nan
        c_end = float(F[1][-1]) if F is not None else np.nan
        b_on = bool(F is not None and np.any(np.abs(F[4] - 1) > 1e-6))
        high_err = float(hi.blend_mode_k - hi.realised_k)
        cls = "coefficients"
        if np.isfinite(state_err) and abs(state_err) > 1.5:
            cls = "state"
        elif bool(hi.rain_win) or ncell > 0:
            cls = "rain"
        elif hi.is_day and np.isfinite(cf_end[i]) and np.isfinite(c_end) and abs(c_end - cf_end[i]) > 0.4:
            cls = "cloud"
        elif hi.is_day and (b_on or (hi.season == "wet" and high_err > 2)):
            cls = "sea_breeze"
        rows.append({"origin": o, "logloss_high": hi.blend_logloss, "logloss_low": lo.blend_logloss,
                     "realised_high": hi.realised_k, "mode_high": hi.blend_mode_k, "realised_low": lo.realised_k,
                     "mode_low": lo.blend_mode_k, "state_err": state_err, "rain_win": bool(hi.rain_win),
                     "rain_cells": ncell, "c_end_forecast": c_end, "goes_cf_end": cf_end[i], "sea_breeze_switch": b_on,
                     "season": hi.season, "is_day": bool(hi.is_day), "sector": hi.sector, "sky_class": hi.sky_class,
                     "class": cls})
        fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})
        x_obs = np.arange(-lead + 1, CFG.sim.horizon_min + 1)
        if sp is not None:
            xs = np.arange(-sp.shape[1] + CFG.sim.horizon_min + 1, CFG.sim.horizon_min + 1)
            for s in sp:
                ax[0].plot(xs, s, color="0.7", lw=0.5, alpha=0.6)
        ax[0].plot(x_obs, t1, color="k", lw=1.5, label="observed 1-min T")
        ax[0].plot(np.arange(1, len(mp) + 1), mp, color="C3", lw=1.5, label="ensemble mean path")
        ax[0].axhline(hi.realised_k, color="C0", ls=":", label=f"settled high {hi.realised_k:.0f} / mode {hi.blend_mode_k:.0f}")
        ax[0].axhline(lo.realised_k, color="C2", ls=":", label=f"settled low {lo.realised_k:.0f} / mode {lo.blend_mode_k:.0f}")
        ax[0].set_ylabel("F")
        ax[0].legend(fontsize=7, loc="best")
        ax[0].set_title(f"{o:%Y-%m-%d %H:%M} UTC  log-loss H {hi.blend_logloss:.2f} L {lo.blend_logloss:.2f}  class: {cls}")
        if F is not None:
            xx = np.arange(1, F.shape[1] + 1)
            ax[1].plot(xx, F[0], label="clear-sky GHI"); ax[1].set_ylabel("W m-2"); ax[1].legend(fontsize=7)
            ax[2].plot(xx, F[1], label="cloud c"); ax[2].plot(xx, F[4], label="b scale")
            ax[2].plot(xx, F[3], label="rain r"); ax[2].legend(fontsize=7)
            ax[3].plot(xx, F[2], label="T_adv"); ax[3].set_ylabel("F"); ax[3].legend(fontsize=7)
        ax[3].set_xlabel("minutes after origin")
        fig.tight_layout()
        fig.savefig(od / f"{i + 1:02d}_{o:%Y%m%dT%H%M}.png", dpi=80)
        plt.close(fig)
    d = pd.DataFrame(rows)
    d.to_csv(od / "classification.csv", index=False)
    return d
