"""Phase 5 diagnostics on a backtest run: innovations, rank histogram, analysis RMSE."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from kmia2h.timeutil import from_ns
from kmia2h.config import CFG, out, store
from kmia2h.data.qc import clean
from kmia2h.filter.process_model import IDX
from kmia2h.target import round_rule


def innovations(tag: str, first: str, last: str) -> pd.DataFrame:
    rows = []
    for p in pd.period_range(first, last, freq="M"):
        d = np.load(store(f"backtest/{tag}/{p.year}{p.month:02d}.npz"))
        z = d["inn_val"] / np.sqrt(d["inn_var"])
        df = pd.DataFrame({"kind": d["inn_kind"], "k": d["inn_k"], "z": z, "month": str(p)})
        rows.append(df)
    d = pd.concat(rows)
    out_ = []
    for kind, g in d.groupby("kind"):
        g = g.sort_values(["month", "k"])
        z = g.z.to_numpy()
        same = (g.month.to_numpy()[1:] == g.month.to_numpy()[:-1])
        ac1 = np.corrcoef(z[1:][same], z[:-1][same])[0, 1]
        out_.append({"kind": kind, "n": len(z), "mean": z.mean(), "var": z.var(), "lag1_ac": ac1})
    return pd.DataFrame(out_)


def ranks(tag: str, first: str, last: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = pd.read_parquet(store("kmia_1min.parquet"), filters=[("ts_utc", ">=", pd.Timestamp(first + "-01", tz="UTC"))])
    t1 = clean(k)
    f30 = pd.read_parquet(store("features_30m.parquet"), columns=["t_last", "season", "is_day"])
    rows = []
    for p in pd.period_range(first, last, freq="M"):
        f = store(f"enkf_snapshots/{tag}_{p.year}{p.month:02d}.npz")
        if not f.exists():
            continue
        d = np.load(f)
        org = from_ns(d["origin"])
        ens = d["ensemble"]
        y = ens[:, :, IDX["T"]] + ens[:, :, IDX["X_slow"]] + ens[:, :, IDX["X_fast"]]
        truth = t1.reindex(org).to_numpy()
        yq = round_rule(y, CFG.target.rounding)
        below = (yq < truth[:, None]).sum(1)
        ties = (yq == truth[:, None]).sum(1)
        rank = below + np.floor(rng.random(len(org)) * (ties + 1)).astype(int)
        ff = f30.reindex(org)
        rows.append(pd.DataFrame({"origin": org, "truth": truth, "ens_mean": y.mean(1), "ens_sd": y.std(1),
                                  "rank": rank, "n": y.shape[1], "t_last": ff.t_last.to_numpy(),
                                  "season": ff.season.to_numpy(), "is_day": ff.is_day.to_numpy()}))
    return pd.concat(rows)


def summary(tag: str, first: str, last: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    inn = innovations(tag, first, last)
    r = ranks(tag, first, last)
    r = r[np.isfinite(r.truth)]
    rh = []
    for (s, dn), g in r.groupby(["season", "is_day"]):
        h, _ = np.histogram(g["rank"], bins=10, range=(0, g.n.iloc[0] + 1))
        chi = stats.chisquare(h)
        rh.append({"season": s, "is_day": dn, "n": len(g), "chi2_p": chi.pvalue,
                   **{f"dec{i}": h[i] / h.sum() for i in range(10)}})
    rh = pd.DataFrame(rh)
    rm = {"rmse_ensemble_mean": float(np.sqrt(np.mean((r.ens_mean - r.truth) ** 2))),
          "rmse_last_obs": float(np.sqrt(np.nanmean((r.t_last - r.truth) ** 2))), "n": int(len(r)),
          "mean_ens_sd": float(r.ens_sd.mean())}
    inn.to_csv(out(5, f"innovations_{tag}.csv"), index=False)
    rh.to_csv(out(5, f"rank_hist_{tag}.csv"), index=False)
    return inn, rh, rm
