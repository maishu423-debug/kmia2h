"""Phase 10: run.py reproduces the stored Test A forecast; outages give a forecast or ABSTAIN, never a
crash; the climate-day composition behaves like a max / min of independent parts."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from kmia2h.config import CFG, out, store
from kmia2h.live import compose

HAVE_STORE = store("obs.parquet").exists() and store(f"backtest/p9_{CFG.frozen.backtest_tag}/202503.npz").exists()
needs_store = pytest.mark.skipif(not HAVE_STORE, reason="needs the stores and the Phase 9 backtest")
ORIGIN = pd.Timestamp("2025-03-02T18:00Z")


# ---- composition (pure) ------------------------------------------------------------------------

def test_combine_high_is_max_of_independent_parts():
    rng = np.random.default_rng(0)
    a = rng.integers(70, 75, 200000)
    b = rng.integers(72, 78, 200000)
    ks_a, p_a = np.arange(70, 75), np.full(5, 0.2)
    ks_b, p_b = np.arange(72, 78), np.full(6, 1 / 6)
    ks, p = compose.combine(ks_a, p_a, ks_b, p_b, "high")
    emp = np.array([(np.maximum(a, b) == k).mean() for k in ks])
    assert np.isclose(p.sum(), 1.0) and np.abs(p - emp).max() < 0.01
    ks, p = compose.combine(ks_a, p_a, ks_b, p_b, "low")
    emp = np.array([(np.minimum(a, b) == k).mean() for k in ks])
    assert np.isclose(p.sum(), 1.0) and np.abs(p - emp).max() < 0.01


def test_combine_dominated_part():
    """An observed high far above the forecast window fixes the day high."""
    ks, p = compose.combine(np.array([90]), np.array([1.0]), np.arange(70, 80), np.full(10, 0.1), "high")
    assert ks[np.argmax(p)] == 90 and np.isclose(p.max(), 1.0)


def test_lst_midnight():
    assert compose.lst_midnight(pd.Timestamp("2026-11-15T04:59Z")) == pd.Timestamp("2026-11-14T05:00Z")
    assert compose.lst_midnight(pd.Timestamp("2026-11-15T05:00Z")) == pd.Timestamp("2026-11-15T05:00Z")


def test_elapsed_pmf_fitted_on_2022_2023():
    pm = compose.load()
    assert pm["fit_period"] == ["2022-01-01", "2023-12-31"]
    for side in ("high", "low"):
        assert np.allclose(np.array(pm[side]).sum(1), 1.0)


# ---- run.py on the stores --------------------------------------------------------------------------

@needs_store
def test_replay_reproduces_stored_test_a_forecast():
    from kmia2h.live import run as live
    f = live.forecast(live.Run(origin=ORIGIN, mode="replay"))
    z = np.load(store(f"backtest/p9_{CFG.frozen.backtest_tag}/{ORIGIN:%Y%m}.npz"))
    i = np.flatnonzero(z["origin"] == ORIGIN.value)[0]
    assert np.array_equal(f["_arrays"]["max_u"], z["max_u"][i])
    assert np.array_equal(f["_arrays"]["min_u"], z["min_u"][i])
    t = pd.read_parquet(out(9, "per_origin_A.parquet"))
    for side, key in (("high", "p_high"), ("low", "p_low")):
        r = t[(t.origin == ORIGIN) & (t.side == side)].iloc[0]
        p = f["_arrays"][key]
        j = int(r.realised_k) - (f[side]["mode"] - int(np.argmax(p)))
        assert -np.log(max(p[j], 1e-4)) == r.blend_logloss
        assert f[side]["mode"] == int(r.blend_mode_k)


@needs_store
@pytest.mark.parametrize("outage", [(), ("goes",), ("mrms",), ("hrrr",), ("kmia",), ("goes", "mrms", "hrrr"),
                                    ("goes", "mrms", "hrrr", "kmia")])
def test_outage_gives_forecast_or_abstain(outage):
    from kmia2h.live import run as live
    f = live.forecast(live.Run(origin=ORIGIN, mode="spinup", outage=outage, n_paths=1000,
                               window_start=compose.lst_midnight(ORIGIN)))
    if "kmia" in outage:
        assert f["status"] == "ABSTAIN" and f["action"] == "ABSTAIN"
        return
    assert f["status"] == "OK"
    for side in ("high", "low"):
        p = np.array(list(f[side]["bins"].values()))
        assert abs(p.sum() - 1) < 0.01 and f[side]["action"] in ("ACT", "ABSTAIN")
        assert abs(sum(f["day"][side]["bins"].values()) - 1) < 0.01
    for k in outage:
        assert any(n.startswith(k) for n in f["degraded"]), (k, f["degraded"])
    json.dumps({k: v for k, v in f.items() if not k.startswith("_")}, default=str)


@needs_store
def test_missing_source_tables_do_not_crash():
    """A live fetch that fails returns None for that source."""
    from kmia2h.live import run as live
    tabs = live.store_tables(ORIGIN - pd.Timedelta(hours=12), ORIGIN)
    for k in ("goes_feat", "mrms_feat", "nwp_point", "ndbc"):
        tabs[k] = None
    f = live.forecast(live.Run(origin=ORIGIN, mode="spinup", tables=tabs, n_paths=500))
    assert f["status"] == "OK" and len(f["degraded"]) == 3
    tabs["obs"] = None
    f = live.forecast(live.Run(origin=ORIGIN, mode="spinup", tables=tabs, n_paths=500))
    assert f["status"] == "ABSTAIN"


@needs_store
def test_no_neighbour_substitution():
    """With KMIA gone, neighbours are present but the run abstains."""
    from kmia2h.live import run as live
    tabs = live.apply_outage(live.store_tables(ORIGIN - pd.Timedelta(hours=12), ORIGIN), ("kmia",), ORIGIN)
    o = tabs["obs"]
    assert (o[o.station != CFG.station].ts_utc > ORIGIN - pd.Timedelta(minutes=20)).any()
    f = live.forecast(live.Run(origin=ORIGIN, mode="spinup", tables=tabs, n_paths=500))
    assert f["status"] == "ABSTAIN"


@needs_store
def test_window_crossing_midnight_uses_truncated_physics():
    from kmia2h.live import run as live
    o = pd.Timestamp("2025-03-03T04:00Z")            # 23:00 LST: one hour of the day left
    f = live.forecast(live.Run(origin=o, mode="spinup", n_paths=500, window_start=compose.lst_midnight(o)))
    assert f["status"] == "OK"
    assert f["day"]["complete"] and f["day"]["forecast_part"].startswith("physics, truncated")
    assert f["day"]["covers_to_utc"] == "2025-03-03T04:59Z"


@pytest.mark.skipif(os.environ.get("KMIA2H_LIVE_TEST") != "1", reason="network; set KMIA2H_LIVE_TEST=1")
def test_origin_now_under_90s(tmp_path):
    import time
    import run as cli
    t = time.time()
    f = cli.main(["--origin", "now", "--window-start", "day", "--out", str(tmp_path / "forecast.json"), "--no-state"])
    assert time.time() - t < 90 and (tmp_path / "forecast.json").exists()
    assert f["status"] in ("OK", "ABSTAIN")
