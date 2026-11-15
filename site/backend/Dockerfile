# Backend image: Django API + paper trading + the kmia2h model (run in process).
# Build context = the REPOSITORY ROOT (Railway: Backend root directory "/", this file via site/backend/railway.json),
# because the backend imports the kmia2h package and reads config.yaml, params/ and the frozen MOS models.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

# ecCodes for HRRR GRIB2 decoding. The locked eccodeslib wheel bundles the library too; the system package
# (and its definitions) is installed as well so findlibs always finds one.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libeccodes0 libeccodes-data ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first (cached layer): the locked kmia2h stack + the site extra (Django, gspread, ...).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra site --no-install-project

# The live bundle (README.md "Live bundle") and the backend.
COPY kmia2h ./kmia2h
COPY config.yaml ./
COPY params ./params
COPY models/mos_high_s9_sb03.joblib models/mos_low_s9_sb03.joblib ./models/
COPY site/backend ./site/backend
RUN uv sync --frozen --no-dev --extra site \
 && python -c "import eccodes; print('eccodes', eccodes.codes_get_api_version())" \
 && python -c "from kmia2h.mos import pipeline; [pipeline.load_model(s, 's9_sb03') for s in ('high', 'low')]; print('MOS models ok')"

WORKDIR /app/site/backend
EXPOSE 8000
# One worker: the forecast, temperature-monitor, price-tracker and autotrader threads must run exactly once.
CMD ["sh", "-c", "python manage.py migrate && gunicorn config.wsgi:application --bind [::]:${PORT:-8000} --workers 1 --threads 8 --timeout 0 --access-logfile -"]
