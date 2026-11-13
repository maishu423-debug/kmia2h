"""CLI: uv run python scripts/run_backtest.py TAG FIRST LAST [--nosim] [--ens] [--overwrite]"""
import sys

from kmia2h.backtest import backtest

tag, first, last = sys.argv[1:4]
flags = set(sys.argv[4:])
backtest.run(tag, first, last, simulate="--nosim" not in flags, save_ensembles="--ens" in flags,
             overwrite="--overwrite" in flags)
