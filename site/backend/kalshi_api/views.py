import hmac
import json
import os

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .algorithm import get_full_state
from .autotrader import get_status as autotrader_status
from .client import KalshiClientError, get_miami_temperature_payload
from .forecast import trigger_refresh, MODELS
from .price_tracker import record_price, nws_round_temp_f
from .paper import PaperTradeError, get_state, parse_order, place_order, reset_account, set_auto_enabled
from .sheets_history import get_recent_snapshots, get_recent_trades
from .stream import stream_events
from .temp_monitor import refresh_temp_now


@require_GET
def health(_request):
    return JsonResponse({"ok": True})


@require_GET
def weather_source_health(request):
    """
    Which live weather source is active and how fresh its newest observation is. Read-only; contains no credentials (only the API host name).
    `?live=1` also pings the LDM API's /health.
    """
    from urllib.parse import urlsplit

    from weather import asos_client, config as wcfg
    from weather.asos_nowcast import freshness_status
    from weather.ldm_client import LdmApi, load_ldm_config, validate_source_config

    try:
        source = wcfg.get_asos_source()
    except wcfg.AsosConfigError:
        source = None
    feed = asos_client.get_feed()
    with feed.lock:
        info = dict(feed.last_poll or {})
        last_good = dict(feed.last_good) if feed.last_good else None
    latency = info.get("latency") or {}
    age = latency.get("observation_age_seconds")
    data = {
        "configured_source": source,
        "config_problems": validate_source_config(),
        "station": wcfg.ASOS_STATION,
        "last_poll": {
            "at": info.get("poll_started_at"),
            "source": info.get("source"),
            "success": info.get("last_poll_success"),
            "source_status": info.get("source_status"),
            "error_category": info.get("source_error_category"),
            "errors": [str(e)[:300] for e in info.get("errors", [])][:10],
            "ldm_listing_ms": info.get("ldm_listing_ms"),
            "ldm_missing_hours": info.get("ldm_missing_hours", []),
            "backfilled_from_public": info.get("backfilled_from_public", []),
            "coverage_missing_hours": info.get("coverage_missing_hours", []),
        },
        "latest_observation": {
            "observed_at": last_good["observed_at"] if last_good else None,
            "temp_f": last_good["temp_f"] if last_good else None,
            "age_seconds": age,
            "freshness": freshness_status(age),
        },
        "latency": latency or None,
    }
    if source == wcfg.ASOS_SOURCE_LDM:
        try:
            conf = load_ldm_config()
            data["ldm_host"] = urlsplit(conf.base_url).netloc
            if request.GET.get("live") == "1":
                data["ldm_health"] = LdmApi(conf).health()
        except wcfg.AsosConfigError:
            pass
    return JsonResponse(data)


@require_GET
def nws_dsm(_request):
    """
    KMIA ASOS Daily Summary Message (latest + last final), decoded, with the raw text, plus `floor`: whether the
    DSM high is currently being used as today's high (it is, when higher than the HFMETAR high).
    """
    from . import nws

    try:
        return JsonResponse({**nws.get_dsm(), "floor": nws.last_floor_decision()})
    except Exception as exc:     # network/parse failure with nothing cached yet
        return JsonResponse({"error": f"Could not load the NWS daily summary (DSM): {type(exc).__name__}"}, status=502)


@require_GET
def miami_temperature(request):
    selected_market_ticker = request.GET.get("market_ticker")

    try:
        payload = get_miami_temperature_payload(selected_market_ticker)
    except KalshiClientError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    return JsonResponse(payload)


@require_GET
def miami_temperature_stream(_request):
    response = StreamingHttpResponse(stream_events(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@require_GET
def paper_state(_request):
    return JsonResponse(get_state())


@csrf_exempt
@require_POST
def paper_order(request):
    try:
        result = place_order(parse_order(request.body))
    except PaperTradeError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse(result)


@csrf_exempt
@require_POST
def paper_reset(_request):
    return JsonResponse(reset_account())


# ── Algorithm endpoints ──────────────────────────────────────────────────────

_ALL_MODEL_NAMES = {m["name"] for m in MODELS}


def _parse_enabled_models(raw: str | None):
    """Parse comma-separated model names; return None if all enabled."""
    if not raw:
        return None
    names = {n.strip() for n in raw.split(",") if n.strip() in _ALL_MODEL_NAMES}
    return names if names else None


@require_GET
def algorithm_state(request):
    """Full algorithm state: forecasts, momentum analysis, recommendation."""
    enabled_models = _parse_enabled_models(request.GET.get("enabled_models"))

    try:
        payload = get_miami_temperature_payload(include_orderbook=False)
    except KalshiClientError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    # Feed current prices into the velocity tracker
    for m in payload["markets"]:
        record_price(m["ticker"], m.get("yes_bid_cents"), m.get("yes_ask_cents"))

    paper = get_state()
    state = get_full_state(payload["markets"], paper, enabled_models=enabled_models)
    return JsonResponse(state)


@require_GET
def autotrader_state(_request):
    """Paper account, positions, trades, bot cycles and schedule status."""
    return JsonResponse({**get_state(), "bot": autotrader_status()})


@csrf_exempt
@require_POST
def autotrader_toggle(request):
    try:
        body = json.loads(request.body.decode()) if request.body else {}
    except json.JSONDecodeError:
        body = {}
    set_auto_enabled(bool(body.get("enabled")))
    return JsonResponse({**get_state(), "bot": autotrader_status()})


@csrf_exempt
@require_POST
def notify_test(_request):
    """Send a test alert to every configured channel so you can check the setup."""
    from .notifier import notify_all

    results = notify_all("Kalshi bot: test alert", "✅ Alerts are working.")
    if not results:
        return JsonResponse({"ok": False, "detail": "No alert channel is configured on the server."}, status=400)
    ok = any(r["ok"] for r in results)
    detail = "; ".join(f"{r['channel']}: {r['detail']}" for r in results)
    return JsonResponse({"ok": ok, "detail": detail, "results": results}, status=200 if ok else 400)


@csrf_exempt
@require_POST
def autotrader_reset(_request):
    reset_account()
    return JsonResponse({**get_state(), "bot": autotrader_status()})


@require_GET
def forecast_history(request):
    """Recent forecast-cycle snapshots: current temp vs. model forecast at each timestamp."""
    limit = int(request.GET.get("limit", 100))
    return JsonResponse({"history": get_recent_snapshots(limit)})


@require_GET
def trade_history(request):
    """Recent paper trades, persisted to Google Sheets (survives Render restarts)."""
    limit = int(request.GET.get("limit", 100))
    return JsonResponse({"history": get_recent_trades(limit)})


@csrf_exempt
@require_POST
def algorithm_refresh(_request):
    """Kick off a manual forecast refresh (runs in background)."""
    trigger_refresh()
    return JsonResponse({"message": "Forecast refresh started."})


@csrf_exempt
@require_POST
def algorithm_cron_refresh(request):
    """Secret-protected forecast refresh for external cron schedulers."""
    expected = os.environ.get("CRON_REFRESH_SECRET", "").strip()
    provided = (
        request.GET.get("secret")
        or request.headers.get("X-Cron-Secret")
        or ""
    ).strip()

    if not expected:
        return JsonResponse({"error": "CRON_REFRESH_SECRET is not configured."}, status=503)
    if not hmac.compare_digest(provided, expected):
        return JsonResponse({"error": "Invalid cron secret."}, status=403)

    trigger_refresh()
    return JsonResponse({"message": "Cron forecast refresh started."})


@csrf_exempt
@require_POST
def algorithm_refresh_temp(_request):
    """Immediately fetch the latest weather reading and return updated temp."""
    asos = refresh_temp_now()
    if asos is None:
        return JsonResponse({"error": "Temperature refresh failed."}, status=502)
    high_f = asos.get("today_high_f")
    if high_f is None:
        high_f = asos.get("t0_f")
    return JsonResponse({
        "current_f":    round(float(asos["t0_f"]), 1),
        "daily_high_f": nws_round_temp_f(high_f),
        "observed_at":  asos.get("observed_at"),
    })
