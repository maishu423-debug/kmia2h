#!/bin/bash
# REBUILD step 5 downloads (resumable). Each job runs until its process exits.
cd "$(dirname "$0")/.."
case "$1" in
  goes) uv run python -m kmia2h.data.fetch_goes 2022-01-01 2024-12-31 ${2:-10} ;;
  hrrr) uv run python -m kmia2h.data.fetch_nwp 2022-01-01 2024-12-31 ${2:-6} ;;
  mrms) uv run python -m kmia2h.data.fetch_mrms 2022-01-01 2024-12-31 ${2:-4} ;;
esac
echo "$1 DONE"
