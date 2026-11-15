#!/usr/bin/env python
"""KMIA 2-hour high / low: one live run of the frozen pipeline (SPEC.md Phase 10).

  run.py --origin now                                  fetch live sources, forecast, write forecast.json
  run.py --origin 2026-11-08T15:39Z                    historical run (replay of the backtest filter)
  run.py --origin now --window-start 2026-11-15T05:00Z compose the Kalshi climate day so far + 2 h forecast
  run.py --origin 2025-03-04T18:00Z --mode spinup --outage goes     outage test (also mrms, hrrr, kmia)

--window-start: start of the climate day, midnight LST = 05:00Z. "day" means the LST midnight that
starts the origin's climate day. The observed part runs from there to the origin.
Writes out/live/forecast.json (or --out) and appends to out/live/runs.jsonl; prints a one-line summary.
"""
import argparse
import json
import sys
import time

import pandas as pd


def parse_time(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s.replace("Z", "+00:00"))
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", required=True, help="'now' or a UTC time, e.g. 2026-11-08T15:39Z")
    ap.add_argument("--window-start", default=None, help="climate-day start (UTC time) or 'day' (LST midnight)")
    ap.add_argument("--mode", choices=("auto", "replay", "spinup", "live"), default="auto",
                    help="auto: live for 'now', replay for a historical origin")
    ap.add_argument("--outage", default="", help="comma list of sources to remove: goes,mrms,hrrr,kmia")
    ap.add_argument("--outage-min", type=int, default=60, help="outage length before the origin (minutes)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--paths", type=int, default=None)
    ap.add_argument("--out", default=None, help="forecast.json path (default out/live/forecast.json)")
    ap.add_argument("--no-state", action="store_true", help="live: ignore the persisted filter state")
    a = ap.parse_args(argv)

    from kmia2h.config import CFG
    from kmia2h.live import compose, run as live
    t0 = time.time()
    now = a.origin.strip().lower() == "now"
    origin = pd.Timestamp.now(tz="UTC").floor("min") if now else parse_time(a.origin).floor("min")
    mode = a.mode if a.mode != "auto" else ("live" if now else "replay")
    data_end = pd.Timestamp(CFG.periods.data_end, tz="UTC") + pd.Timedelta(days=1)
    if mode in ("replay", "spinup") and origin >= data_end:
        sys.exit(f"origin {origin} is after the stored data (data_end {CFG.periods.data_end}); use --mode live")
    ws = None
    if a.window_start:
        ws = compose.lst_midnight(origin) if a.window_start == "day" else parse_time(a.window_start)
    r = live.Run(origin=origin, mode=mode, window_start=ws,
                 outage=tuple(s for s in a.outage.split(",") if s), outage_min=a.outage_min,
                 seed=a.seed, n_paths=a.paths, use_state=not a.no_state)
    if mode == "live":
        from kmia2h.live import fetch
        r.tables, r.fetch_log = fetch.fetch_all(origin)
    f = live.forecast(r)
    f.setdefault("timing", {})["wall_s"] = round(time.time() - t0, 1)
    live.write(f, a.out)
    print(f["summary"])
    return f


if __name__ == "__main__":
    main()
