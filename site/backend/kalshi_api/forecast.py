"""
Background forecast runner: the kmia2h model (this repository), called as a library every FORECAST_REFRESH_SECONDS.

Each cycle:
  1. KMIA HFMETAR = the observations the temperature monitor has already downloaded (the shared feed's cache;
     the forecast never triggers a second NOAA / LDM download for them).
  2. Every other live input from kmia2h's own fetchers: KMIA routine METARs (aviationweather.gov and IEM),
     neighbour stations (IEM), buoys (NDBC), GOES, MRMS and HRRR.
  3. One pass of the frozen kmia2h pipeline (kmia2h.live.run.forecast) at origin = now, with the climate day
     (midnight to midnight LST) as the window start.
  4. Three entries in the JSON shape the algorithm and the page read (T0, temps/probs, high_temps/high_probs,
     mean, std, mode, p10, p90):
        kmia2h_high_2h   settled high of the next 2 hours
        kmia2h_low_2h    settled low of the next 2 hours
        kmia2h_day_high  climate-day high: the observed part of the day composed with the forecast
     The algorithm trades on kmia2h_day_high only.

When the HFMETAR feed has no usable current observation, nothing runs: the last forecast is kept, the loop reports
"waiting for HFMETAR" and retries every HFMETAR_RETRY_SECONDS. When the model ABSTAINs (no KMIA report in the last
20 minutes), the three entries carry {"error": "ABSTAIN: ..."} and the algorithm waits.
"""
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone

MODELS = [
    {"name": "kmia2h_high_2h", "label": "2-h high", "kind": "high_2h"},
    {"name": "kmia2h_low_2h", "label": "2-h low", "kind": "low_2h"},
    {"name": "kmia2h_day_high", "label": "Day high", "kind": "day_high"},
]
DAY_HIGH_MODEL = "kmia2h_day_high"

REFRESH_INTERVAL = int(os.getenv("FORECAST_REFRESH_SECONDS", "300"))   # seconds (default 5 minutes)
HFMETAR_RETRY_SECONDS = int(os.getenv("HFMETAR_RETRY_SECONDS", "30"))
STARTUP_DELAY = int(os.getenv("FORECAST_STARTUP_DELAY_SECONDS", "30"))  # let /ping/ respond and the monitor poll first
FEED_REUSE_SECONDS = 15 * 60       # read the monitor's cache; never contact NOAA / LDM from here while it is running
HFMETAR_LOOKBACK_HOURS = 9         # 6 h filter spin-up + trend features; the cache holds what the monitor fetched
KNOTS_PER_MPH = 1 / 1.150779
# cloud_fraction (max over layers, weather/asos_parser.py) -> the dominant METAR cover code the model reads
COVER_FROM_FRACTION = ((0.0, "CLR"), (0.2, "FEW"), (0.45, "SCT"), (0.75, "BKN"), (1.0, "OVC"))

_lock = threading.Lock()
_started = False
_loop = {
    "started": False,
    "started_at": None,
    "last_cycle_started_at": None,
    "last_cycle_finished_at": None,
    "next_run_at": None,
    "last_error": None,
    "waiting_for_hfmetar": None,   # {"reason": str, "since": iso} while HFMETAR is unavailable
}
_cache = {
    "results": {},       # model name -> entry (or {"error": ...})
    "last_run_at": None,  # naive UTC ISO (the autotrader reads it that way)
    "running": False,
    "asos_current": None,
    "run": None,         # summary of the last kmia2h run (status, timing, degraded sources)
}


# ── inputs ───────────────────────────────────────────────────────────────────────────────────────

def _cover_code(fraction):
    if fraction is None:
        return None
    return min(COVER_FROM_FRACTION, key=lambda fc: abs(fc[0] - float(fraction)))[1]


def hfmetar_table(observations, now):
    """The monitor's HFMETAR observations as rows of the model's observation table (KMIA, report_type 1).

    available_at = observation time + the HFMETAR latency the model was fitted with (config.yaml latency_min), the
    same rule as in training; a report is never used before it exists."""
    import pandas as pd
    from kmia2h.config import CFG
    from kmia2h.data.parse_obs import COLS

    lat = pd.Timedelta(minutes=CFG.latency_min.hfmetar)
    rows = []
    for o in observations or []:
        if o.get("temp_f") is None:
            continue
        ts = pd.Timestamp(o["observed_at"]).tz_convert("UTC")
        if ts > now:
            continue
        sp = o.get("wind_speed_mph")
        rows.append({"station": CFG.station, "ts_utc": ts, "report_type": 1, "tmpf": float(o["temp_f"]),
                     "tmpc_tenths": float("nan"),
                     "dwpf": o.get("dewpoint_f"), "sknt": None if sp is None else sp * KNOTS_PER_MPH,
                     "drct": o.get("wind_dir_deg"), "skyc1": _cover_code(o.get("cloud_fraction")),
                     "wxcodes": o.get("weather"), "p01i": None, "synthetic": False, "available_at": ts + lat})
    d = pd.DataFrame(rows).reindex(columns=COLS)
    if len(d):
        d["report_type"] = d["report_type"].astype("int8")
        for c in ("ts_utc", "available_at"):
            d[c] = d[c].astype("datetime64[ns, UTC]")
        for c in ("skyc1", "skyc2", "skyc3", "skyc4", "wxcodes"):
            d[c] = d[c].astype("string")
        d["synthetic"] = d["synthetic"].astype(bool)
    return d


def _merge_obs(hf, fetched):
    """KMIA HFMETAR from the monitor + everything else the kmia2h fetch returned (its own KMIA HFMETAR rows dropped)."""
    import pandas as pd
    from kmia2h.config import CFG
    parts = [hf]
    if fetched is not None and len(fetched):
        parts.append(fetched[~((fetched.station == CFG.station) & (fetched.report_type == 1))])
    d = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
    for c in ("ts_utc", "available_at"):
        d[c] = d[c].astype("datetime64[ns, UTC]")
    return d.drop_duplicates(["station", "ts_utc", "report_type"]).sort_values(["station", "ts_utc"]).reset_index(drop=True)


def _current_from_feed(observations, now):
    """asos_current for the price tracker and the page: latest HFMETAR temperature and the provable recorded high."""
    from weather.asos_nowcast import todays_high

    from .nws import resolve_known_high_detail
    latest = observations[-1]
    high, high_raw = resolve_known_high_detail(todays_high(observations, now)["high_f"], now)
    return {"t0_f": float(latest["temp_f"]), "today_high_f": high, "today_high_raw_f": high_raw,
            "observed_at": latest["observed_at"]}


def _apply_asos_current(asos: dict):
    """Push the latest KMIA reading into the shared temp tracker."""
    if not asos:
        return
    with _lock:
        _cache["asos_current"] = asos
    from .price_tracker import set_daily_high_nws, set_daily_high_raw, update_temp
    if asos.get("t0_f") is not None:
        update_temp(float(asos["t0_f"]))
    if asos.get("today_high_f") is not None:
        set_daily_high_nws(float(asos["today_high_f"]))
    if asos.get("today_high_raw_f") is not None:
        set_daily_high_raw(float(asos["today_high_raw_f"]))


# ── output shape ─────────────────────────────────────────────────────────────────────────────────

def summary_entry(bins: dict, t0, fetched_at, extra=None) -> dict:
    """{"85": 0.4, ...} -> T0, temps/probs, high_temps/high_probs, mean, std, mode, mode_prob, p10, p90."""
    pairs = sorted((int(k), float(v)) for k, v in bins.items())
    total = sum(p for _, p in pairs) or 1.0
    temps = [t for t, _ in pairs]
    probs = [round(p / total, 5) for _, p in pairs]
    mean = sum(t * p for t, p in zip(temps, probs))
    std = math.sqrt(max(0.0, sum(p * (t - mean) ** 2 for t, p in zip(temps, probs))))
    mode_i = max(range(len(probs)), key=probs.__getitem__)
    cdf, p10, p90 = 0.0, None, None
    for t, p in zip(temps, probs):
        cdf += p
        if p10 is None and cdf >= 0.10:
            p10 = t
        if p90 is None and cdf >= 0.90:
            p90 = t
    return {"T0": None if t0 is None else round(float(t0), 2), "temps": temps, "probs": probs,
            "high_temps": temps, "high_probs": probs, "mean": round(mean, 2), "std": round(std, 3),
            "mode": temps[mode_i], "mode_prob": probs[mode_i], "p10": p10, "p90": p90,
            "fetched_at": fetched_at, "fit_source": "kmia2h", **(extra or {})}


def results_from_forecast(f: dict) -> dict:
    """kmia2h forecast dict -> {model name: entry}."""
    fetched = f.get("origin_utc")
    if f.get("status") != "OK":
        err = {"error": f"ABSTAIN: {f.get('reason', 'no forecast')}", "fetched_at": fetched}
        return {m["name"]: dict(err) for m in MODELS}
    t0 = f.get("latest_kmia_f")
    common = {"window_utc": f.get("window_utc"), "regime": f.get("regime"), "degraded": f.get("degraded", [])}
    out = {}
    for name, side in (("kmia2h_high_2h", "high"), ("kmia2h_low_2h", "low")):
        s = f[side]
        out[name] = summary_entry(s["bins"], t0, fetched, {
            **common, "label": next(m["label"] for m in MODELS if m["name"] == name),
            "action": s["action"], "p_correct": s["p_correct"], "edge": s["edge"]})
    day = f.get("day")
    if day:
        out[DAY_HIGH_MODEL] = summary_entry(day["high"]["bins"], t0, fetched, {
            **common, "label": "Day high", "window_start_utc": day["window_start_utc"],
            "covers_to_utc": day["covers_to_utc"], "complete": day["complete"],
            "forecast_part": day["forecast_part"], "observed_high_f": day.get("report_max_f")})
    else:
        out[DAY_HIGH_MODEL] = {"error": "no climate-day composition", "fetched_at": fetched}
    return out


# ── one cycle ────────────────────────────────────────────────────────────────────────────────────

def run_model(now=None):
    """One kmia2h pass. Returns (results, run summary) or (None, reason) when HFMETAR is unavailable."""
    import pandas as pd
    from weather.asos_client import fetch_recent_observations
    from weather.asos_nowcast import STATUS_UNAVAILABLE, freshness_status
    from weather.asos_parser import parse_utc
    from weather.climate_day import climate_day_start

    now = now or datetime.now(timezone.utc)
    observations, _info = fetch_recent_observations(lookback_hours=HFMETAR_LOOKBACK_HOURS, now=now,
                                                    reuse_seconds=FEED_REUSE_SECONDS)
    observations = [o for o in observations or [] if o.get("temp_f") is not None]
    if not observations:
        return None, "no HFMETAR observations"
    age = (now - parse_utc(observations[-1]["observed_at"])).total_seconds()
    if freshness_status(age) == STATUS_UNAVAILABLE:
        return None, f"newest HFMETAR is {age / 60:.0f} min old"
    _apply_asos_current(_current_from_feed(observations, now))

    from kmia2h.live import fetch as live_fetch, run as live
    origin = pd.Timestamp(now).tz_convert("UTC").floor("min")
    tables, log = live_fetch.fetch_all(origin)
    tables["obs"] = _merge_obs(hfmetar_table(observations, origin), tables.get("obs"))
    log["hfmetar_from_monitor"] = f"{len(observations)} observations"
    window_start = pd.Timestamp(climate_day_start(origin.to_pydatetime()))
    f = live.forecast(live.Run(origin=origin, mode="live", window_start=window_start, tables=tables, fetch_log=log))
    run = {"status": f["status"], "summary": f.get("summary"), "origin_utc": f.get("origin_utc"),
           "timing": f.get("timing"), "degraded": f.get("degraded", []), "filter": f.get("filter"),
           "sources": log}
    return results_from_forecast(f), run


def _do_refresh():
    """Run the model once. True = ran, False = HFMETAR unavailable (retry soon), None = already running."""
    with _lock:
        if _cache["running"]:
            return None
        _cache["running"] = True
    try:
        results, run = run_model()
    except Exception as exc:
        with _lock:
            _cache["running"] = False
            _loop["last_error"] = f"{type(exc).__name__}: {exc}"[:400]
        print(f"[forecast] kmia2h run failed ({type(exc).__name__}: {exc})", flush=True)
        return True
    if results is None:
        with _lock:
            waiting = _loop["waiting_for_hfmetar"]
            _loop["waiting_for_hfmetar"] = {"reason": f"weather_unavailable: {run}",
                                            "since": waiting["since"] if waiting else datetime.utcnow().isoformat()}
            _cache["running"] = False
        if waiting is None:
            print(f"[forecast] waiting for HFMETAR ({run}); retrying every {HFMETAR_RETRY_SECONDS}s", flush=True)
        return False
    with _lock:
        _loop["waiting_for_hfmetar"] = None
        _cache["results"] = results
        _cache["run"] = run
        _cache["last_run_at"] = datetime.utcnow().isoformat()
        _cache["running"] = False
    print(f"[forecast] {run.get('summary', '').splitlines()[0] if run.get('summary') else run['status']}", flush=True)
    _record_forecast_snapshot(results)
    return True


def _record_forecast_snapshot(results: dict):
    """Persist a history row: current temp + the day-high forecast at this cycle (local CSV + Google Sheet)."""
    try:
        from .algorithm import get_ensemble_forecast
        from .forecast_log import append_forecast
        from .price_tracker import get_temp_snapshot
        from .sheets_history import append_snapshot

        ensemble = get_ensemble_forecast(results)
        snap = get_temp_snapshot()
        if not ensemble or snap.get("current_f") is None:
            return
        append_forecast(ensemble, snap["current_f"], snap.get("daily_high_f"))
        append_snapshot(current_temp_f=snap["current_f"], model_forecast_f=ensemble.get("mean"))
    except Exception as exc:
        print(f"[forecast] failed to record history snapshot ({exc})", flush=True)


def _background_loop():
    with _lock:
        _loop["started"] = True
        _loop["started_at"] = datetime.utcnow().isoformat()
        _loop["next_run_at"] = (datetime.utcnow() + timedelta(seconds=STARTUP_DELAY)).isoformat()
    time.sleep(STARTUP_DELAY)
    while True:
        started_at = time.time()
        with _lock:
            _loop["last_cycle_started_at"] = datetime.utcnow().isoformat()
            _loop["next_run_at"] = None
            _loop["last_error"] = None
        interval = REFRESH_INTERVAL
        try:
            if _do_refresh() is False:
                interval = HFMETAR_RETRY_SECONDS
        except Exception as exc:
            with _lock:
                _loop["last_error"] = str(exc)
            print(f"[forecast] background refresh crashed ({exc}); will retry in {REFRESH_INTERVAL}s", flush=True)
        sleep_for = max(0, interval - (time.time() - started_at))
        with _lock:
            _loop["last_cycle_finished_at"] = datetime.utcnow().isoformat()
            _loop["next_run_at"] = (datetime.utcnow() + timedelta(seconds=sleep_for)).isoformat()
        time.sleep(sleep_for)


def start_background_refresh():
    """Call once at Django startup (AppConfig.ready)."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_background_loop, daemon=True, name="forecast-bg").start()


def trigger_refresh():
    """Fire-and-forget manual refresh (used by the API)."""
    threading.Thread(target=_do_refresh, daemon=True, name="forecast-manual").start()


def get_forecast_cache():
    """A snapshot of the cache (thread-safe copy)."""
    with _lock:
        return {
            "results": dict(_cache["results"]),
            "last_run_at": _cache["last_run_at"],
            "running": _cache["running"],
            "asos_current": _cache["asos_current"],
            "run": _cache["run"],
            "loop": dict(_loop),
        }
