"""kalshi_api/forecast.py: the kmia2h model as the only forecast source (BUILD_SITE.md)."""
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalshi_api import forecast  # noqa: E402

KEYS = ("T0", "temps", "probs", "high_temps", "high_probs", "mean", "std", "mode", "p10", "p90")


def _fc(status="OK"):
    side = lambda m: {"bins": {str(m - 1): 0.2, str(m): 0.5, str(m + 1): 0.3}, "action": "ABSTAIN",  # noqa: E731
                      "p_correct": 0.4, "edge": 0.3}
    return {"status": status, "reason": "no KMIA HFMETAR / METAR in the last 20 min", "origin_utc": "2026-11-15T17:00Z",
            "latest_kmia_f": 80.6, "window_utc": ["17:01Z", "19:00Z"], "regime": "day-onshore-sct", "degraded": [],
            "high": side(83), "low": side(79),
            "day": {"window_start_utc": "2026-11-15T05:00Z", "covers_to_utc": "2026-11-15T19:00Z", "complete": False,
                    "forecast_part": "final model (blend)", "report_max_f": 82.4,
                    "high": {"bins": {"83": 0.6, "84": 0.4}}, "low": {"bins": {"70": 1.0}}}}


def test_three_entries_in_the_shape_the_algorithm_and_page_read():
    res = forecast.results_from_forecast(_fc())
    assert list(res) == [m["name"] for m in forecast.MODELS] == ["kmia2h_high_2h", "kmia2h_low_2h", "kmia2h_day_high"]
    for name, e in res.items():
        assert all(k in e for k in KEYS), name
        assert sum(e["probs"]) == pytest.approx(1.0) and e["T0"] == 80.6
    hi = res["kmia2h_high_2h"]
    assert hi["temps"] == [82, 83, 84] and hi["mode"] == 83 and hi["mean"] == pytest.approx(83.1)
    assert hi["p10"] == 82 and hi["p90"] == 84 and hi["std"] == pytest.approx(0.7, abs=1e-3)
    day = res["kmia2h_day_high"]
    assert day["high_temps"] == [83, 84] and day["complete"] is False and day["covers_to_utc"] == "2026-11-15T19:00Z"


def test_abstain_gives_errors_and_the_algorithm_waits(monkeypatch):
    from kalshi_api import algorithm as alg
    res = forecast.results_from_forecast(_fc("ABSTAIN"))
    assert all(e["error"].startswith("ABSTAIN") for e in res.values())
    monkeypatch.setattr(alg, "get_temp_snapshot", lambda: {"current_f": None, "daily_high_f": None})
    live = [{"ticker": "A", "label": "A", "floor": 80, "cap": 81, "yes_ask_cents": 40, "no_ask_cents": 61,
             "is_dead": False, "model_prob": None}]
    assert alg.get_recommendation(live, results=res)["status"] == "waiting"


def test_hfmetar_rows_follow_the_model_schema():
    now = datetime(2026, 11, 15, 17, 0, tzinfo=timezone.utc)
    obs = [{"observed_at": "2026-11-15T16:50:00+00:00", "temp_f": 80.6, "dewpoint_f": 68.0, "wind_speed_mph": 11.507,
            "wind_dir_deg": 90.0, "cloud_fraction": 0.45, "weather": None},
           {"observed_at": "2026-11-15T17:05:00+00:00", "temp_f": 81.0}]                       # in the future: dropped
    d = forecast.hfmetar_table(obs, now)
    assert len(d) == 1
    r = d.iloc[0]
    assert r.station == "KMIA" and r.report_type == 1 and r.tmpf == 80.6 and r.skyc1 == "SCT"
    assert r.sknt == pytest.approx(10.0, abs=1e-3)
    assert (r.available_at - r.ts_utc).total_seconds() == 8 * 60     # the latency the model was fitted with


def test_monitor_hfmetar_replaces_the_fetched_kmia_hfmetar():
    import pandas as pd
    now = datetime(2026, 11, 15, 17, 0, tzinfo=timezone.utc)
    hf = forecast.hfmetar_table([{"observed_at": "2026-11-15T16:50:00+00:00", "temp_f": 80.6}], now)
    other = forecast.hfmetar_table([{"observed_at": "2026-11-15T16:45:00+00:00", "temp_f": 99.0}], now)
    metar = other.assign(report_type=3, tmpf=80.1, ts_utc=pd.Timestamp("2026-11-15T16:53Z"))
    nb = other.assign(station="KOPF", tmpf=81.0)
    d = forecast._merge_obs(hf, pd.concat([other, metar, nb], ignore_index=True))
    kmia = d[d.station == "KMIA"]
    assert sorted(kmia.report_type.tolist()) == [1, 3] and 99.0 not in kmia.tmpf.tolist()
    assert (d.station == "KOPF").sum() == 1


def test_the_forecast_reads_the_monitor_cache(monkeypatch):
    """No second LDM / NOAA download: the feed is asked to reuse its last poll."""
    import weather.asos_client as ac
    seen = {}

    def fake(lookback_hours=None, now=None, session=None, reuse_seconds=0):
        seen["reuse"] = reuse_seconds
        return [], {}
    monkeypatch.setattr(ac, "fetch_recent_observations", fake)
    assert forecast.run_model() == (None, "no HFMETAR observations")
    assert seen["reuse"] >= 10 * 60


STORE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "store", "obs.parquet")


@pytest.mark.skipif(not os.path.exists(STORE), reason="needs the kmia2h stores")
def test_one_full_cycle_offline(monkeypatch, tmp_path):
    """A whole cycle at a historical origin: the monitor's HFMETAR (from the store) + store tables in place of the
    live fetchers -> three entries, and the day-high entry drives the algorithm."""
    import pandas as pd
    import weather.asos_client as ac
    from kmia2h.live import fetch as live_fetch, run as live
    from kalshi_api import algorithm as alg

    origin = pd.Timestamp("2024-07-15T19:00Z")
    tabs = live.store_tables(origin - pd.Timedelta(hours=24), origin)
    k = tabs["obs"][(tabs["obs"].station == "KMIA") & (tabs["obs"].report_type == 1)]
    feed = [{"observed_at": t.isoformat(), "temp_f": float(v)} for t, v in zip(k.ts_utc, k.tmpf)]
    monkeypatch.setattr(ac, "fetch_recent_observations", lambda **kw: (feed, {}))
    monkeypatch.setattr(live_fetch, "fetch_all", lambda o: (dict(tabs), {"seconds": 0}))
    monkeypatch.setattr(live, "STATE_FILE", tmp_path / "state.npz")
    monkeypatch.setattr(forecast, "_current_from_feed", lambda obs, now: {"t0_f": obs[-1]["temp_f"]})
    res, run = forecast.run_model(now=origin.to_pydatetime())
    assert run["status"] == "OK", run
    assert set(res) == {"kmia2h_high_2h", "kmia2h_low_2h", "kmia2h_day_high"}
    for e in res.values():
        assert all(key in e for key in KEYS) and sum(e["probs"]) == pytest.approx(1.0, abs=1e-3)
    assert res["kmia2h_day_high"]["window_start_utc"] == "2024-07-15T05:00Z"
    assert res["kmia2h_day_high"]["mode"] >= res["kmia2h_high_2h"]["p10"]
    assert alg._get_day_high_pdf(res)
