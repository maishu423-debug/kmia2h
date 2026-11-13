"""NWS daily climate reports (CLI) for KMIA -> store/cli.parquet (date, high, low, final)."""
from __future__ import annotations

import time

import pandas as pd
import requests

from kmia2h.config import CFG, store


def main(first: int = 2016, last: int = 2026) -> pd.DataFrame:
    rows = []
    for y in range(first, last + 1):
        r = requests.get(f"{CFG.iem.base}/json/cli.py", params={"station": CFG.station, "year": y},
                         timeout=CFG.iem.timeout_s)
        r.raise_for_status()
        for x in r.json()["results"]:
            issued = pd.Timestamp(x["product"][:8]) if x.get("product") else pd.NaT
            rows.append({"date": pd.Timestamp(x["valid"]).date(), "high": x["high"], "low": x["low"],
                         "product": x.get("product"),
                         # The final report is issued the morning after; a same-day product is preliminary.
                         "final": bool(pd.notna(issued) and issued.date() > pd.Timestamp(x["valid"]).date())})
        time.sleep(CFG.iem.min_interval_s + 1)
    d = pd.DataFrame(rows)
    for c in ("high", "low"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.drop_duplicates("date", keep="last").sort_values("date")
    d.to_parquet(store("cli.parquet"), index=False)
    print(d.shape, d.final.mean())
    return d


if __name__ == "__main__":
    main()
