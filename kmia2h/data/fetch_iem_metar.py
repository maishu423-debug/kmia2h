"""METAR (routine+specials) and HFMETAR for KMIA and neighbours -> raw monthly chunks."""
from __future__ import annotations

import sys

from kmia2h.config import CFG
from kmia2h.data.iem import fetch_chunk, month_chunks


def iem_id(station: str) -> str:
    return station[1:] if len(station) == 4 and station.startswith("K") else station


def download(stations: list[str], start: str, end: str, report_types=("1", "3", "4")) -> None:
    for st in stations:
        for a, b in month_chunks(start, end):
            params = [("station", iem_id(st)), ("data", "all"), *[("report_type", r) for r in report_types],
                      ("tz", "UTC"), ("format", "onlycomma"), ("latlon", "no"), ("missing", "empty"),
                      ("trace", "0.0001")]
            fetch_chunk("metar", "/cgi-bin/request/asos.py", params, f"{st}_{a:%Y%m}", a, b)
            print("metar", st, a.strftime("%Y-%m"), flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    st = [CFG.station] if which == "kmia" else list(CFG.neighbours) if which == "neigh" else which.split(",")
    download(st, sys.argv[2], sys.argv[3])
