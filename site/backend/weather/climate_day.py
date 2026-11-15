"""The KMIA climate day: midnight to midnight local STANDARD time (EST, UTC-5) all year.

The NWS climate report, the DSM and the Kalshi KXHIGHMIA market all use this day. ASOS clocks do not follow daylight
saving, so in summer 00:00-01:00 EDT still belongs to the PREVIOUS climate day. Wall-clock schedules (the trading
window, the 4 PM cutoff) are Eastern time and use zoneinfo America/New_York instead; never mix the two."""
from datetime import date, datetime, timedelta, timezone

LST = timezone(timedelta(hours=-5), "LST")


def climate_date(now: datetime | None = None) -> date:
    """The climate day `now` falls in."""
    now = now or datetime.now(timezone.utc)
    return now.astimezone(LST).date()


def climate_day_start(now: datetime | None = None) -> datetime:
    """UTC start (LST midnight = 05:00Z) of the climate day containing `now`."""
    d = climate_date(now)
    return datetime(d.year, d.month, d.day, tzinfo=LST).astimezone(timezone.utc)


def climate_day_end(now: datetime | None = None) -> datetime:
    return climate_day_start(now) + timedelta(days=1)
