"""Raw IEM METAR/HFMETAR chunks -> store/obs.parquet (SPEC.md Phase 2 step 3).

IEM's CSV does not carry the report type. HFMETAR (5-minute) reports carry the MADISHF tag in
the raw text; everything else is a routine or special METAR."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from kmia2h.config import CFG, path, store

TGROUP = re.compile(r"\sT([01])(\d{3})([01])(\d{3})")
MAINGROUP = re.compile(r"\s(M?\d{2})/(M?\d{2})?\s")
COLS = ["station", "ts_utc", "report_type", "tmpf", "tmpc_tenths", "dwpf", "sknt", "drct", "skyc1", "skyc2",
        "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "wxcodes", "p01i", "synthetic", "available_at"]


def parse_tgroup(metar: str) -> tuple[float, float]:
    """T group -> (T, Td) in degC to tenths. Sign digit 1 = negative."""
    m = TGROUP.search(" " + str(metar))
    if not m:
        return np.nan, np.nan
    t = int(m.group(2)) / 10.0 * (-1 if m.group(1) == "1" else 1)
    d = int(m.group(4)) / 10.0 * (-1 if m.group(3) == "1" else 1)
    return t, d


def parse_main(metar: str) -> tuple[float, float]:
    """Whole-degC T/Td group (e.g. 27/24, M05/M10)."""
    m = MAINGROUP.search(" " + str(metar) + " ")
    if not m:
        return np.nan, np.nan

    def v(s):
        if not s:
            return np.nan
        return -float(s[1:]) if s.startswith("M") else float(s)
    return v(m.group(1)), v(m.group(2))


def c_to_f(c):
    return np.asarray(c, float) * 9.0 / 5.0 + 32.0


def parse_file(f, station: str) -> pd.DataFrame:
    d = pd.read_csv(f, low_memory=False, na_values=["M", ""], keep_default_na=True)
    if d.empty:
        return pd.DataFrame(columns=COLS)
    d["ts_utc"] = pd.to_datetime(d["valid"], utc=True)
    metar = d["metar"].astype(str)
    hf = metar.str.contains("MADISHF", regex=False)
    tg = np.array([parse_tgroup(m) for m in metar])
    mg = np.array([parse_main(m) for m in metar])
    o = pd.DataFrame({"station": station, "ts_utc": d["ts_utc"]})
    o["report_type"] = np.where(hf, 1, 3).astype(np.int8)
    o["tmpc_tenths"] = np.where(hf, np.nan, tg[:, 0])
    # HFMETAR: whole degC only. Routine: tenths from the T group, else whole degC, else IEM tmpf.
    t_c = np.where(hf, mg[:, 0], np.where(np.isnan(tg[:, 0]), mg[:, 0], tg[:, 0]))
    td_c = np.where(hf, mg[:, 1], np.where(np.isnan(tg[:, 1]), mg[:, 1], tg[:, 1]))
    o["tmpf"] = np.where(np.isnan(t_c), pd.to_numeric(d.get("tmpf"), errors="coerce"), c_to_f(t_c))
    o["dwpf"] = np.where(np.isnan(td_c), pd.to_numeric(d.get("dwpf"), errors="coerce"), c_to_f(td_c))
    for c in ("sknt", "drct", "p01i", "skyl1", "skyl2", "skyl3", "skyl4"):
        o[c] = pd.to_numeric(d.get(c), errors="coerce")
    for c in ("skyc1", "skyc2", "skyc3", "skyc4", "wxcodes"):
        o[c] = d.get(c).astype("string") if c in d else pd.Series(pd.NA, index=d.index, dtype="string")
    o["synthetic"] = False
    lat = CFG.latency_min
    lag = np.where(station == CFG.station, np.where(hf, lat.hfmetar, lat.metar), lat.neighbour)
    o["available_at"] = o["ts_utc"] + pd.to_timedelta(lag, unit="min")
    return o[COLS]


def build_raw() -> pd.DataFrame:
    parts = []
    for f in sorted(path("raw", "iem", "metar").glob("*.csv.gz")):
        st = f.name.split("_")[0]
        parts.append(parse_file(f, st))
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(["station", "ts_utc", "report_type"]).sort_values(["station", "ts_utc"])
    return d.reset_index(drop=True)


def synth_hfmetar(k1: pd.DataFrame, offset: int = 0) -> pd.Series:
    """ASOS-style 5-minute report from 1-minute data: rolling 5-minute mean at minutes ending 0/5
    -> round to degF -> whole degC -> degF. offset shifts which 1-minute block is averaged."""
    from kmia2h.data.qc import clean
    from kmia2h.target import rolling_mean, round_rule
    rule = CFG.target.rounding
    t = clean(k1)
    r = pd.Series(rolling_mean(t.to_numpy()), index=t.index).shift(offset)
    r = r[r.index.minute % 5 == 0]
    f = round_rule(r.to_numpy(), rule)
    c = round_rule((f - 32.0) * 5.0 / 9.0, rule)
    return pd.Series(c_to_f(c), index=r.index)


def build(k1: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    d = build_raw()
    if k1 is None:
        k1 = pd.read_parquet(store("kmia_1min.parquet"))
    kh = d[(d.station == CFG.station) & (d.report_type == 1)].set_index("ts_utc")["tmpf"]
    kh = kh[~kh.index.duplicated()]
    # Choose the block alignment that matches real HFMETAR best.
    stats = {}
    for off in (0, 1):
        s = synth_hfmetar(k1, off)
        j = pd.concat([kh.rename("real"), s.rename("syn")], axis=1, join="inner").dropna()
        stats[off] = float((np.isclose(j.real, j.syn)).mean()) if len(j) else np.nan
    best = max(stats, key=lambda k: -1 if np.isnan(stats[k]) else stats[k])
    syn = synth_hfmetar(k1, best).dropna()
    # Months with < 90% HFMETAR coverage get synthetic reports where real ones are missing.
    obs_end = d.loc[d.station == CFG.station, "ts_utc"].max()
    syn = syn[syn.index <= obs_end]
    pm = syn.index.tz_localize(None).to_period("M")
    expected = pd.Series(pm.unique().days_in_month * 288, index=pm.unique())
    cov = (kh.groupby(kh.index.tz_localize(None).to_period("M")).size().reindex(expected.index).fillna(0) / expected)
    low = set(cov[cov < CFG.features.hfmetar_synth_min_cov].index)
    take = syn[pm.isin(low) & ~syn.index.isin(kh.index)]
    s = pd.DataFrame({"station": CFG.station, "ts_utc": take.index, "report_type": np.int8(1), "tmpf": take.values,
                      "synthetic": True})
    s["available_at"] = s["ts_utc"] + pd.Timedelta(minutes=CFG.latency_min.hfmetar)
    d = pd.concat([d, s.reindex(columns=COLS)], ignore_index=True).sort_values(["station", "ts_utc"])
    d["synthetic"] = d["synthetic"].fillna(False).astype(bool)
    d.reset_index(drop=True).to_parquet(store("obs.parquet"), index=False)
    info = {"match_by_offset": stats, "offset": best, "synth_months": sorted(str(m) for m in low),
            "hf_cov": cov}
    return d, info
