"""CLI: uv run python scripts/run_backtest.py TAG FIRST LAST [--nosim] [--ens] [--overwrite]"""
import sys

from kmia2h.backtest import backtest

tag, first, last = sys.argv[1:4]
flags = set(a for a in sys.argv[4:] if not a.startswith("--sources="))
src = [a.split("=", 1)[1] for a in sys.argv[4:] if a.startswith("--sources=")]
src = tuple(src[0].split(",")) if src and src[0] else ()
backtest.run(tag, first, last, simulate="--nosim" not in flags, save_ensembles="--ens" in flags,
             overwrite="--overwrite" in flags, sources=src)
