"""BUILD_SITE.md bug fixes: the 4 PM cutoff uses America/New_York (not a hard-coded UTC-4), and the climate day is
midnight to midnight local STANDARD time all year (00:00-01:00 EDT in summer belongs to the previous day)."""
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weather import asos_nowcast  # noqa: E402
from weather.climate_day import climate_date, climate_day_start  # noqa: E402

SUMMER_0030_EDT = datetime(2026, 7, 15, 4, 30, tzinfo=timezone.utc)     # 00:30 EDT on 15 Jul = 23:30 EST on 14 Jul
SUMMER_0130_EDT = datetime(2026, 7, 15, 5, 30, tzinfo=timezone.utc)
WINTER_0030_EST = datetime(2026, 1, 15, 5, 30, tzinfo=timezone.utc)


def test_cutoff_uses_new_york_wall_clock_in_both_seasons():
    from kalshi_api.algorithm import hours_to_cutoff
    assert hours_to_cutoff(datetime(2026, 7, 15, 19, 0, tzinfo=timezone.utc)) == pytest.approx(1.0)   # 15:00 EDT
    assert hours_to_cutoff(datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)) == pytest.approx(1.0)   # 15:00 EST
    assert hours_to_cutoff(datetime(2026, 1, 15, 21, 30, tzinfo=timezone.utc)) == 0.0                 # after 4 PM EST


def test_climate_day_is_lst_all_year():
    assert climate_date(SUMMER_0030_EDT).isoformat() == "2026-07-14"
    assert climate_date(SUMMER_0130_EDT).isoformat() == "2026-07-15"
    assert climate_date(WINTER_0030_EST).isoformat() == "2026-01-15"
    assert climate_day_start(SUMMER_0130_EDT) == datetime(2026, 7, 15, 5, 0, tzinfo=timezone.utc)


def _obs(t, f):
    return {"observed_at": t.isoformat().replace("+00:00", "Z"), "temp_f": f}


def test_recorded_high_counts_00_to_01_edt_toward_the_previous_day():
    late = datetime(2026, 7, 15, 4, 40, tzinfo=timezone.utc)     # 00:40 EDT: still the 14th's climate day
    early = datetime(2026, 7, 15, 5, 10, tzinfo=timezone.utc)    # 01:10 EDT: the 15th
    obs = [_obs(late, 88.0), _obs(early, 80.6)]
    today = asos_nowcast.todays_high(obs, SUMMER_0130_EDT)
    assert today["miami_date"] == "2026-07-15" and today["high_f"] == 80.6
    yesterday = asos_nowcast.todays_high(obs[:1], SUMMER_0030_EDT + (late - SUMMER_0030_EDT))
    assert yesterday["miami_date"] == "2026-07-14" and yesterday["high_f"] == 88.0


def test_dsm_for_the_climate_day_matches_after_midnight_edt(monkeypatch):
    from kalshi_api import nws
    monkeypatch.setattr(nws, "_dsm_fail_until", 0.0)
    entry = {"kind": "partial", "date": "2026-07-14", "high_f": 91, "issued_at": "2026-07-15T04:20:00+00:00"}
    monkeypatch.setattr(nws, "get_dsm", lambda: {"latest": entry})
    got, why = nws._todays_dsm_high(SUMMER_0030_EDT)
    assert got is entry and why == "ok"
    got, why = nws._todays_dsm_high(SUMMER_0130_EDT)          # an hour later the 14th's DSM is yesterday's
    assert got is None and "not today" in why


def test_market_of_the_day_follows_the_climate_day():
    from kalshi_api.client import get_today_event_ticker
    assert get_today_event_ticker(SUMMER_0030_EDT) == "KXHIGHMIA-26JUL14"
    assert get_today_event_ticker(SUMMER_0130_EDT) == "KXHIGHMIA-26JUL15"
