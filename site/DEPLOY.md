# Railway deployment — KMIA live site

The site deploys as two Railway services from **this one repository**: a Django `Backend` (API, temperature monitor,
the kmia2h forecast every 5 minutes, paper-trading bot) and a Next.js `Frontend`. Forecasts come only from the kmia2h
model in this repository; there is no AccuWeather or other forecast provider.

## 1. Create the services

Create an empty Railway project and add two services connected to this GitHub repository:

| Service | Root directory | Config file | Builder |
| --- | --- | --- | --- |
| `Backend` | `/` (the repository root) | `/site/backend/railway.json` | Dockerfile `site/backend/Dockerfile` |
| `Frontend` | `/site/frontend` | `/site/frontend/railway.json` | Railpack / Nixpacks (Node) |

- Set each service's config-file path explicitly. Railway resolves config-file paths from the repository root, not
  from the service's root directory.
- The Backend's root directory **must be the repository root**. Its image needs the `kmia2h` package,
  `config.yaml`, `params/` and the two frozen MOS models (`models/mos_{high,low}_s9_sb03.joblib`) next to
  `site/backend`. The root `.dockerignore` limits the build context to exactly those files: no training data,
  stores or logs.
- The Backend image (`python:3.12-slim-bookworm`) installs ecCodes (`libeccodes0`, `libeccodes-data`; the locked
  `eccodeslib` wheel bundles it as well) for HRRR GRIB2 decoding. The build fails if `import eccodes` or loading the
  MOS models fails.
- Keep the Backend at **one replica** (`numReplicas: 1` in its railway.json) and one Gunicorn worker. The SQLite
  database, the filter state and the background threads are single-instance.
- Do not enable Serverless / app sleeping for the Backend: sleeping stops the forecast, temperature-monitor,
  price-tracker and autotrader threads.
- Memory: a forecast cycle (fetch + filter + 4,000 simulated paths + MOS) peaks at roughly 1–1.5 GB. Give the Backend
  at least 2 GB.

## 2. Add persistent Backend storage

Attach a Railway volume to `Backend` with this mount path:

```text
/data
```

It holds the SQLite database (paper account, positions, trades, bot runs), the HFMETAR last-good observation, the
kmia2h filter state and the local forecast log. Migrations are part of the start command because Railway volumes are
not mounted in pre-deploy containers.

## 3. Configure Backend variables

Secrets go **only** in Railway service variables, never in the repository, a Frontend variable (`NEXT_PUBLIC_*`), a
log or a screenshot. The backend never returns a secret from any endpoint.

Required:

```env
PORT=8000
DJANGO_DEBUG=false
DJANGO_SECRET_KEY=<long random value>
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,.railway.app,.railway.internal,healthcheck.railway.app
DATABASE_PATH=/data/db.sqlite3
RUN_BACKGROUND_TASKS=true
TZ=UTC
```

Persistent state on the volume:

```env
ASOS_STATE_PATH=/data/asos_last_good.json          # last good HFMETAR observation
KMIA2H_STATE_FILE=/data/kmia2h_filter_state.npz    # kmia2h filter state between 5-minute runs
FORECAST_LOG_PATH=/data/forecast_log.csv           # one row per forecast cycle (local copy of the history)
```

Live weather source (KMIA HFMETAR). Same names as the old project:

```env
ASOS_SOURCE=ldm                                    # ldm | public   (unset = public NOAA files)
LDM_API_BASE_URL=https://data.getedgefront.com
LDM_API_TOKEN=<secret>
```

Optional tuning (defaults shown):

```env
LDM_API_TIMEOUT_SECONDS=15
LDM_API_MAX_RETRIES=2
LDM_BACKFILL_PUBLIC=true                           # hours the LDM API no longer holds come once from NOAA's files
MADIS_POLL_MINUTES=1                               # temperature-monitor cadence
ASOS_FRESH_MINUTES=15
ASOS_MAX_CURRENT_AGE_MINUTES=25
ASOS_ARRIVAL_LOG_PATH=                             # e.g. /data/logs/asos_arrivals.jsonl (one line per new report)
```

`ASOS_SOURCE=ldm` without `LDM_API_TOKEN` stops the deploy at `python manage.py migrate` with check `weather.E001`.
There is no silent fallback. Roll back to NOAA's public files by setting `ASOS_SOURCE=public` (no rebuild).

Forecast loop (defaults shown):

```env
FORECAST_REFRESH_SECONDS=300                       # the kmia2h model every 5 minutes
HFMETAR_RETRY_SECONDS=30                           # retry cadence while HFMETAR is unavailable
FORECAST_STARTUP_DELAY_SECONDS=30
```

The forecast reads KMIA HFMETAR from the temperature monitor's cache, so it never downloads from LDM / NOAA itself.
Its other inputs are public and need no keys: IEM (neighbour stations, KMIA routine METARs), aviationweather.gov
(KMIA METARs), NDBC (buoys), and the NOAA GOES, MRMS and HRRR buckets on AWS (anonymous).

Kalshi. Market data is public, so the page and the paper bot work without keys. Keys only enable the authenticated
WebSocket price stream:

```env
KALSHI_API_KEY_ID=
KALSHI_PRIVATE_KEY_PEM=                            # the PEM itself; real newlines or escaped \n both work
KALSHI_API_BASE_URL=https://external-api.kalshi.com/trade-api/v2   # optional; the demo URL to rehearse
```

Use `KALSHI_PRIVATE_KEY_PEM`, not a key path. Nothing in this site sends real orders: the bot trades the paper
account only (100 contracts per leg, YES on the most likely live bucket, NO on every other, every 15 minutes from
12:30 to 15:30 ET, 98¢ caps, held to settlement). The copied profit watcher is not started.

Google Sheets history. This project uses **new** sheets: create two new Google Sheets, share both with the
service-account email (Editor), and set:

```env
GOOGLE_SERVICE_ACCOUNT_JSON=<the whole service-account JSON on one line>
GOOGLE_SHEETS_ID=<NEW forecast-history sheet id>
GOOGLE_TRADES_SHEET_ID=<NEW trade-history sheet id>
```

Alerts (each channel is optional; the bot alerts every configured channel each cycle):

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=
EMAIL_APP_PASSWORD=
EMAIL_TO=
```

Optional secret-protected refresh for an external scheduler (`POST /api/kalshi/algorithm/cron-refresh/`):

```env
CRON_REFRESH_SECRET=<long random value>
```

## 4. Configure Frontend variables

```env
PORT=3000
NODE_ENV=production
BACKEND_URL=http://${{Backend.RAILWAY_PRIVATE_DOMAIN}}:${{Backend.PORT}}
```

`Backend` in the reference must exactly match the Backend service name. `BACKEND_URL` is read only by the Next.js
server-side rewrite (`/api/*` → Backend), so the Backend can stay private. Generate a public Railway domain for
`Frontend` only.

## 5. Verify

1. Backend health check (Railway runs it on deploy): `/api/kalshi/health/` → `{"ok": true}`.
2. Through the Frontend's rewrite: `https://YOUR-FRONTEND-DOMAIN/api/kalshi/health/`.
3. Weather source: `https://YOUR-FRONTEND-DOMAIN/api/kalshi/weather/source/`. Check that `configured_source` is
   `ldm`, `config_problems` is empty and `latest_observation.freshness` is `healthy`.
4. About 1 minute after a deploy (30 s start-up delay, then fetch + model ≈ 10–20 s), the backend log prints one
   `[forecast] high: … | low: … | regime … | ACT / ABSTAIN` line per cycle, and
   `https://YOUR-FRONTEND-DOMAIN/api/kalshi/algorithm/state` shows `forecast.models` with `kmia2h_high_2h`,
   `kmia2h_low_2h` and `kmia2h_day_high`, each with `T0`, `temps`/`probs`, `high_temps`/`high_probs`, `mean`, `std`,
   `mode`, `p10`, `p90`.
   - A model ABSTAIN (no KMIA report in the last 20 minutes) shows as `"error": "ABSTAIN: …"` on all three, and the
     bot waits.
   - `forecast.run.degraded` lists any live source (GOES, MRMS, HRRR) that was stale and its fallback.
5. The page shows the three cards (2-h high, 2-h low, Day high), the DSM panel, the probability distribution, bracket
   analysis, forecast history, paper trading and trade history.

## 6. Updating the model

The Backend image bakes in the live bundle (`config.yaml`, `params/`, `models/*.joblib`). After retraining, update
the bundle as described in the repository README ("Live bundle"), push, and Railway rebuilds the Backend (its watch
patterns include `kmia2h/**`, `params/**`, `models/**` and `config.yaml`). The filter state on the volume is reused
if it is less than 6 hours old; otherwise the model spins up from the last 6 hours of observations by itself.

The paper-order, paper-reset, toggle and refresh POST endpoints are not authenticated. Do not add live-money order
execution without adding authentication and authorization first.
