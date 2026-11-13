#!/bin/bash
# Sequential IEM download job (politeness: one request at a time). Resumable.
set -e
cd "$(dirname "$0")/.."
uv run python -m kmia2h.data.fetch_iem_1min download 2016-01-01 2025-12-31
uv run python -m kmia2h.data.fetch_iem_metar kmia 2016-01-01 2024-12-31
uv run python -m kmia2h.data.fetch_iem_metar neigh 2016-01-01 2024-12-31
echo IEM_DONE
