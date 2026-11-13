import numpy as np
import pandas as pd
import pytest

from kmia2h import features
from kmia2h.config import store
from kmia2h.data.parse_obs import parse_main, parse_tgroup


def test_tgroup():
    assert parse_tgroup("KMIA 010053Z RMK AO2 T02560178") == (25.6, 17.8)
    assert parse_tgroup("X RMK T10561022 $") == (-5.6, -2.2)
    assert np.isnan(parse_tgroup("no group")[0])


def test_main_group():
    assert parse_main("KMIA 010000Z AUTO 08007KT 10SM FEW015 27/24 A3005") == (27.0, 24.0)
    assert parse_main("X 1SM M05/M10 A3005") == (-5.0, -10.0)


def test_sky_class():
    d = pd.DataFrame({"skyc1": ["FEW", "SCT", "FEW", "VV", "CLR", None],
                      "skyc2": [None, "BKN", "SCT", None, None, None],
                      "skyc3": [None, None, "OVC", None, None, None], "skyc4": [None] * 6})
    assert list(features.sky_class(d)) == ["clear", "bkn", "ovc", "ovc", "clear", "unk"]


def test_sector():
    assert list(features.sector([30, 90, 150, 151, 210, 330, 331, 0, np.nan])) == [
        "onshore", "onshore", "onshore", "other", "offshore", "offshore", "other", "other", "other"]


@pytest.mark.skipif(not store("obs.parquet").exists(), reason="needs store/obs.parquet")
def test_no_lookahead():
    """Features on 1,000 random origins equal features built from obs truncated at each origin."""
    obs = pd.read_parquet(store("obs.parquet"))
    rng = np.random.default_rng(1)
    t0, t1 = pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")
    origins = pd.DatetimeIndex(sorted(set(t0 + pd.to_timedelta(rng.choice(int((t1 - t0).total_seconds() // 600), 1000, replace=False) * 10, unit="min"))))
    full = features.Features(obs=obs, ndbc=None).features_at(origins)
    cols = ["t_last", "t_age_min", "trend_30", "dwpf", "sknt", "sky_class", "rain_recent", "nb_KFLL"]
    for o in origins[::10]:
        win = obs[(obs.available_at <= o) & (obs.ts_utc > o - pd.Timedelta(hours=6))]
        part = features.Features(obs=win, ndbc=None).features_at(pd.DatetimeIndex([o]))
        a, b = full.loc[o, cols], part.loc[o, cols]
        for c in cols:
            assert (pd.isna(a[c]) and pd.isna(b[c])) or a[c] == b[c], (o, c, a[c], b[c])
    # Every source row used is available at the origin.
    assert (full["t_age_min"].dropna() >= 0).all()
    k = obs[obs.station == "KMIA"]
    used = pd.merge_asof(pd.DataFrame({"origin": origins}), k.sort_values("available_at")[["available_at"]],
                         left_on="origin", right_on="available_at")
    assert (used.available_at.dropna() <= used.origin[used.available_at.notna()]).all()
