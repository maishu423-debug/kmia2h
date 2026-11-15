"""
Model-driven paper-trading strategy.

FORECAST:     The kmia2h model (kalshi_api/forecast.py) gives three distributions: the 2-hour high, the 2-hour
              low and the climate-day high (the observed part of the day composed with the forecast). The
              daily-high PDF used for trading is the climate-day high, floored at the proven recorded high.
EACH CYCLE:   Rank the live Kalshi buckets by model probability.
              YES: on the single highest-probability live bucket.
              NO:  on every other live bucket.
              No comparison with Kalshi's prices: they are only the fill price
              (a leg is skipped if it has no ask or the ask is above the cap).
NO SELLING:   Contracts are held until Kalshi settles the market.
"""
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from weather.asos_resolution import lower_bound_whole_f

from .forecast import DAY_HIGH_MODEL, MODELS, get_forecast_cache
from .price_tracker import get_all_velocities, get_temp_snapshot, nws_round_temp_f


MIAMI_TZ          = ZoneInfo("America/New_York")   # wall clock: EDT in summer, EST in winter
CUTOFF_LOCAL_HOUR = 16

# Never pay more than this for a contract (above it there is no upside left).
NO_MAX_PRICE_CENTS = 98
YES_MAX_PRICE_CENTS = 98

def hours_to_cutoff(now: datetime | None = None) -> float:
    """Hours until 4 PM Eastern wall-clock time (zoneinfo handles EDT / EST)."""
    local = (now or datetime.now(timezone.utc)).astimezone(MIAMI_TZ)
    local_h = local.hour + local.minute / 60.0 + local.second / 3600.0
    return max(0.0, CUTOFF_LOCAL_HOUR - local_h)


def _nws_known_high_f(*values) -> int | None:
    """
    The highest whole-degree-F value we can prove the day has reached, from readings in F. A whole-degree-C reading
    (the NOAA 5-minute feed) only narrows the true value to one or two whole-F values (87.8F is 87 or 88), so the
    lowest is used: a bucket that is still alive is never marked dead. Readings with tenths, and whole-F bounds
    (NWS hourly / DSM), are exact. See weather/asos_resolution.py.
    """
    bounds = [lower_bound_whole_f(v) for v in values if v is not None]
    return max(bounds) if bounds else None


# ── Daily-high PDF: the kmia2h climate-day high ────────────────────────────────

def _get_day_high_pdf(results: dict) -> dict:
    """{temp: prob} from the climate-day high entry ("high_temps"/"high_probs"); {} if it is missing or failed."""
    data = results.get(DAY_HIGH_MODEL) or {}
    if "error" in data or "high_temps" not in data or "high_probs" not in data:
        return {}
    pdf: dict[int, float] = defaultdict(float)
    for t, p in zip(data["high_temps"], data["high_probs"]):
        pdf[int(t)] += float(p)
    norm = sum(pdf.values())
    return {t: p / norm for t, p in pdf.items()} if norm > 0 else {}


def _floor_pdf_at(pdf: dict, known_high_f: float | None) -> dict:
    """The final high can't be below what has already been observed: fold mass below it onto it."""
    if known_high_f is None:
        return dict(pdf)
    h_now = nws_round_temp_f(known_high_f)
    out: dict[int, float] = defaultdict(float)
    for t, p in pdf.items():
        out[max(int(t), h_now)] += p
    return dict(out)


def _compute_daily_high_pdf(known_high_f: float | None, day_high_pdf: dict | None) -> dict:
    """P(final daily high = k): the climate-day high PDF, floored at the proven recorded high."""
    return _floor_pdf_at(day_high_pdf, known_high_f) if day_high_pdf else {}


# ── Step 3: bracket probability from daily-high PDF ──────────────────────────

def _in_bracket(t, floor, cap) -> bool:
    """
    Does integer temperature t settle a Kalshi bucket YES?
      between (floor, cap):  floor <= t <= cap   ("87° to 88°" is floor 87, cap 88)
      greater (floor only):  t > floor           ("89° or above" is floor 88)
      less    (cap only):    t < cap             ("80° or below" is cap 81)
    """
    if floor is not None and cap is not None:
        return float(floor) <= t <= float(cap)
    if floor is not None:
        return t > float(floor)
    if cap is not None:
        return t < float(cap)
    return True


def _bracket_prob(floor, cap, high_pdf: dict) -> float:
    """P(final_daily_high settles the bucket) using integer temperatures."""
    return round(sum(p for t, p in high_pdf.items() if _in_bracket(t, floor, cap)), 4)


# ── Market analysis ───────────────────────────────────────────────────────────

def analyze_markets(markets: list) -> list:
    temp         = get_temp_snapshot()
    current_f    = temp["current_f"]
    daily_high_f = temp["daily_high_f"]
    known        = _nws_known_high_f(current_f, daily_high_f) or 0

    tickers    = [m["ticker"] for m in markets]
    velocities = get_all_velocities(tickers)

    out = []
    for market in markets:
        status = market.get("status", "")
        if status and status not in ("open", "active"):
            continue

        ticker  = market["ticker"]
        floor   = market.get("floor_strike")
        cap     = market.get("cap_strike")
        yes_ask = market.get("yes_ask_cents")
        yes_bid = market.get("yes_bid_cents")
        no_ask  = market.get("no_ask_cents")
        no_bid  = market.get("no_bid_cents")

        if yes_ask is None and yes_bid is None:
            continue

        if no_ask is None and yes_bid is not None:
            no_ask = 100 - yes_bid

        # The daily high only rises: a bucket is dead once the known high is above its top.
        is_dead = (
            (cap is not None and known > (float(cap) if floor is not None else float(cap) - 1))
            or (yes_ask is not None and yes_ask <= 3)
        )

        vels    = velocities.get(ticker, {})
        yes_vel = vels.get("yes_velocity", 0.0)
        no_vel  = vels.get("no_velocity",  0.0)

        out.append({
            "ticker":        ticker,
            "label":         market.get("label", ""),
            "floor":         floor,
            "cap":           cap,
            "yes_ask_cents": yes_ask,
            "yes_bid_cents": yes_bid,
            "no_ask_cents":  no_ask,
            "no_bid_cents":  no_bid,
            "volume":        market.get("volume"),
            "yes_velocity":  round(yes_vel, 3),
            "no_velocity":   round(no_vel,  3),
            "is_dead":       is_dead,
            "model_prob":    None,  # filled in by get_recommendation
        })

    out.sort(key=lambda x: x["is_dead"])
    for i, row in enumerate(out):
        row["rank"] = i + 1

    return out


# ── Recommendation ────────────────────────────────────────────────────────────

def get_recommendation(analyses: list, results=None) -> dict:
    """
    Strategy run each cycle. It does NOT compare the model with Kalshi's prices:
      1. Rank live brackets by model probability.
      2. YES on the highest-probability live bracket.
      3. NO on every other live bracket.
    Kalshi's prices are only used as the price paid. A leg is skipped only when
    there is no ask to fill at, or the ask is above the price cap (no upside left).

    Return shape:
      {
        "status":     "ready" | "waiting",
        "reason":     str,
        "mean":       float | None,   # ensemble mean daily high
        "yes_trade":  {ticker, label, side, price_cents, model_prob, kalshi_prob} | None,
        "no_trades":  [{ticker, label, side, price_cents, model_prob, kalshi_prob}, ...],
      }
    `kalshi_prob` is Kalshi's implied probability that the bracket settles YES (informational only).
    """
    if results is None:
        results = get_forecast_cache().get("results", {})

    temp     = get_temp_snapshot()
    known_hi = _nws_known_high_f(temp.get("current_f"), temp.get("daily_high_f"))

    high_pdf     = _compute_daily_high_pdf(known_hi, _get_day_high_pdf(results))
    has_models   = bool(high_pdf)

    def waiting(reason, mean=None):
        return {"status": "waiting", "reason": reason, "mean": mean, "yes_trade": None, "no_trades": []}

    live = [a for a in analyses if not a["is_dead"]]
    for bracket in live:
        bracket["model_prob"] = (
            _bracket_prob(bracket.get("floor"), bracket.get("cap"), high_pdf)
            if has_models else None
        )

    if not live:
        return waiting("All brackets eliminated by daily high.")
    if not has_models:
        return waiting("Waiting for model forecasts to load.")

    mean_high = round(sum(t * p for t, p in high_pdf.items()), 2)

    def leg(bracket, side, price_cents):
        model_p = bracket["model_prob"] or 0.0
        kalshi_p = price_cents / 100.0 if side == "yes" else 1.0 - price_cents / 100.0
        return {
            "ticker":      bracket["ticker"],
            "label":       bracket["label"],
            "side":        side,
            "price_cents": price_cents,
            "model_prob":  model_p,
            "kalshi_prob": round(kalshi_p, 4),
        }

    top = max(live, key=lambda b: b["model_prob"] or 0.0)
    yes_trade = None
    yes_ask = top.get("yes_ask_cents")
    if yes_ask and yes_ask <= YES_MAX_PRICE_CENTS:
        yes_trade = leg(top, "yes", yes_ask)

    no_trades = []
    for b in live:
        no_ask = b.get("no_ask_cents")
        if b["ticker"] == top["ticker"] or not no_ask or no_ask > NO_MAX_PRICE_CENTS:
            continue
        no_trades.append(leg(b, "no", no_ask))

    if yes_trade is None and not no_trades:
        return waiting(
            f"No tradable price: top bucket {top['label']} (model {(top['model_prob'] or 0.0) * 100:.0f}%) "
            f"and the other live buckets have no ask at or below {YES_MAX_PRICE_CENTS}¢.",
            mean_high,
        )

    parts = []
    if yes_trade:
        parts.append(f"YES {yes_trade['label']} (model {yes_trade['model_prob'] * 100:.0f}%)")
    if no_trades:
        parts.append(f"NO x{len(no_trades)}: " + ", ".join(t["label"] for t in no_trades))

    return {
        "status":    "ready",
        "reason":    f"Forecast {mean_high:.1f}°F — " + " | ".join(parts),
        "mean":      mean_high,
        "yes_trade": yes_trade,
        "no_trades": no_trades,
    }


def _compute_ensemble_stats(results: dict, known_hi: float | None, current_temp_f: float | None) -> dict:
    """Daily-high forecast stats (mean/mode/p10/p90/dist): the climate-day high, floored at the recorded high."""
    high_pdf      = _compute_daily_high_pdf(known_hi, _get_day_high_pdf(results))
    if not high_pdf:
        return {}

    sorted_temps = sorted(high_pdf.keys())
    mean_high    = sum(t * high_pdf[t] for t in sorted_temps)
    mode_high    = max(high_pdf, key=high_pdf.get)
    cdf_val, p10, p90 = 0.0, None, None
    for t in sorted_temps:
        cdf_val += high_pdf[t]
        if p10 is None and cdf_val >= 0.10:
            p10 = t
        if p90 is None and cdf_val >= 0.90:
            p90 = t

    n_valid = sum(
        1 for name, data in results.items()
        if "error" not in data and "temps" in data
    )
    return {
        "current_temp_f": round(current_temp_f or 0, 1),
        "mean":           round(mean_high, 1),
        "mode":           mode_high,
        "mode_prob":      round(high_pdf[mode_high], 4),
        "p10":            p10,
        "p90":            p90,
        "n_models":       n_valid,
        "dist":           [{"temp": t, "prob": round(high_pdf[t], 4)} for t in sorted_temps],
    }


def get_ensemble_forecast(results=None) -> dict:
    """
    Ensemble daily-high forecast summary, independent of any specific markets.
    Used by the background forecast refresh loop to record history snapshots.
    """
    if results is None:
        results = get_forecast_cache().get("results", {})
    temp     = get_temp_snapshot()
    known_hi = _nws_known_high_f(temp.get("current_f"), temp.get("daily_high_f"))
    return _compute_ensemble_stats(results, known_hi, temp.get("current_f"))


# ── Master state ──────────────────────────────────────────────────────────────

def get_full_state(markets: list, paper_state=None, enabled_models=None) -> dict:
    cache       = get_forecast_cache()
    all_results = cache.get("results", {})
    cutoff_h    = hours_to_cutoff()
    nws_temp    = get_temp_snapshot()   # pure NWS — used for display only
    temp        = dict(nws_temp)        # algorithm copy, augmented below with model T0s

    model_t0s = []
    for data in all_results.values():
        if isinstance(data, dict) and "error" not in data and data.get("T0") is not None:
            try:
                model_t0s.append(float(data["T0"]))
            except (TypeError, ValueError):
                pass
    if model_t0s:
        observed_high = max([temp.get("daily_high_f") or 0.0, *model_t0s])
        temp = {**temp, "daily_high_f": observed_high}  # augmented for algorithm only

    # Filter to enabled models for recommendation and ensemble
    results = (
        {k: v for k, v in all_results.items() if k in enabled_models}
        if enabled_models is not None
        else all_results
    )

    # current_f is set by AccuWeather (every 15 min in _do_refresh, or on manual refresh).
    # NWS ASOS only advances daily_high_f — never sets current_f (Celsius-rounding issue).
    # Do not call update_temp() here — model T0s are already folded in via model_t0s above.

    analyses       = analyze_markets(markets)
    recommendation = get_recommendation(analyses, results=results)

    # Per-model summary — always show all cached models in UI
    models_summary = {}
    for name, data in all_results.items():
        if "error" in data:
            models_summary[name] = {"error": data["error"]}
        else:
            probs = data.get("probs", [])
            temps = data.get("temps", [])
            models_summary[name] = {
                **{k: data.get(k) for k in ("label", "window_utc", "action", "p_correct", "edge", "regime",
                                            "degraded", "window_start_utc", "covers_to_utc", "complete",
                                            "forecast_part", "observed_high_f") if k in data},
                "T0":         data.get("T0"),
                "mean":       data.get("mean"),
                "std":        data.get("std"),
                "mode":       data.get("mode"),
                "mode_prob":  round(max(probs), 4) if probs else None,
                "p10":        data.get("p10"),
                "p90":        data.get("p90"),
                "dist":       [
                    {"temp": int(t), "prob": round(float(p), 4)}
                    for t, p in zip(temps, probs)
                ],
                "fetched_at": data.get("fetched_at"),
                "fit_source": data.get("fit_source"),
            }

    # Ensemble stats derived from the daily-high PDF (using enabled models only)
    known_hi = _nws_known_high_f(temp.get("current_f"), temp.get("daily_high_f"))
    ensemble = _compute_ensemble_stats(results, known_hi, temp.get("current_f"))

    h_rem = int(cutoff_h)
    m_rem = int((cutoff_h - h_rem) * 60)
    recorded_high = nws_round_temp_f(nws_temp["daily_high_f"]) if nws_temp["daily_high_f"] is not None else None
    raw_high = nws_temp.get("daily_high_raw_f")
    # What the PAGE shows as the recorded high: the hottest reading (5-minute feed and NWS hourly reports) rounded
    # half-up (80.6 -> 81), but never below a value we have proven (an exact NWS hourly / DSM whole-degree high).
    # Display only: the bot keeps using `recorded_high`, the lowest provable value.
    if recorded_high is None or raw_high is None:
        display_high = recorded_high
    else:
        display_high = max(recorded_high, nws_round_temp_f(raw_high))

    return {
        "forecast": {
            "models":      models_summary,
            "model_order": [m["name"] for m in MODELS],
            "run":         cache.get("run"),
            "ensemble":    ensemble,
            "last_run_at": cache.get("last_run_at"),
            "running":     cache.get("running", False),
            "loop":        cache.get("loop"),
        },
        "temp": {
            "current_f":    round(nws_temp["current_f"],    1) if nws_temp["current_f"]    is not None else None,
            "daily_high_f": recorded_high,
            "daily_high_display_f": display_high,
            # the hottest reading today before rounding, shown next to the recorded high and labelled "feed" on the page.
            # The 5-minute feed is whole degrees C, so it can sit a degree away from the (exact / lowest provable) high.
            "daily_high_raw_f": round(nws_temp["daily_high_raw_f"], 2) if nws_temp.get("daily_high_raw_f") is not None else None,
        },
        "analysis":       analyses,
        "recommendation": recommendation,
        "time_to_cutoff": {
            "hours":   round(cutoff_h, 2),
            "display": f"{h_rem}h {m_rem}m remaining" if cutoff_h > 0 else "Market closed",
        },
        "models_ready": bool(ensemble),
    }
