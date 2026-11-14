#!/bin/bash
# Phase 9 test-period downloads (Test A 2025, Test B 2026 to date). Resumable. Fitting never touches these.
cd "$(dirname "$0")/.."
END=${END:-2026-11-12}
case "$1" in
  iem)
    uv run python -m kmia2h.data.fetch_iem_1min download 2026-01-01 $END
    uv run python -m kmia2h.data.fetch_iem_metar kmia 2025-01-01 $END
    uv run python -m kmia2h.data.fetch_iem_metar neigh 2025-01-01 $END ;;
  goes) uv run python -m kmia2h.data.fetch_goes 2025-01-01 $END ${2:-10} ;;
  hrrr) uv run python -m kmia2h.data.fetch_nwp 2025-01-01 $END ${2:-6} ;;
  mrms) uv run python -m kmia2h.data.fetch_mrms 2025-01-01 $END ${2:-4} ;;
esac
echo "$1 DONE"
