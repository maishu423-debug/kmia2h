"""Polite IEM client: one request at a time, >= min_interval_s apart, exponential backoff on
429/503, chunk halving on 422. Raw responses are cached as gzip CSV per chunk, which doubles as the
resumable manifest of completed keys."""
from __future__ import annotations

import gzip
import time
from pathlib import Path

import pandas as pd
import requests

from kmia2h.config import CFG, path

_last = [0.0]


class Unprocessable(Exception):
    pass


def _get(url: str, params: list[tuple[str, str]]) -> str:
    c = CFG.iem
    for attempt in range(c.retries):
        wait = c.min_interval_s - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(url, params=params, timeout=c.timeout_s)
        except requests.RequestException:
            time.sleep(2 ** (attempt + 1))
            continue
        if r.status_code == 422:
            raise Unprocessable(r.text[:200])
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** (attempt + 2))
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError(f"IEM request failed after {c.retries} attempts: {url} {params}")


def month_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.Timestamp(start, tz="UTC").to_period("M").to_timestamp().tz_localize("UTC")
    e = pd.Timestamp(end, tz="UTC")
    out = []
    while s <= e:
        n = s + pd.offsets.MonthBegin(1)
        out.append((s, min(n, e + pd.Timedelta(days=1))))
        s = n
    return out


def fetch_chunk(kind: str, endpoint: str, params: list[tuple[str, str]], key: str,
                sts: pd.Timestamp, ets: pd.Timestamp) -> Path:
    """Cached download of [sts, ets). Halves the chunk on 422."""
    f = path("raw", "iem", kind, f"{key}.csv.gz")
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return f
    url = CFG.iem.base + endpoint
    fmt = "%Y-%m-%dT%H:%MZ"

    def get(a, b):
        try:
            return _get(url, params + [("sts", a.strftime(fmt)), ("ets", b.strftime(fmt))])
        except Unprocessable:
            mid = a + (b - a) / 2
            mid = mid.floor("h")
            t1, t2 = get(a, mid), get(mid, b)
            return t1 + "\n".join(t2.splitlines()[1:]) + "\n"

    text = get(sts, ets)
    tmp = f.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as g:
        g.write(text)
    tmp.rename(f)
    return f
