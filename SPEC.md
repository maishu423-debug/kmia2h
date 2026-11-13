# KMIA 2-Hour High/Low Model — Build Plan

Nov 9, 2026 · @Y

## For the coding agent — read this first

This doc is an executable spec. Implement Phases 1–10 in order without asking the user anything: every open choice has a default here, and anything still unspecified is yours to decide and log.

**Operating rules**

1. Work one phase at a time. A phase is finished only when its **Done when** checks pass. Then run the full test suite, commit (`phase N: <summary>`), and append the phase's key numbers to `REPORT.md`.
2. Do not ask the user questions. When this doc leaves a choice open, take the simplest option consistent with it, record it in `DECISIONS.md` (date, phase, decision, reason), and continue.
3. Never invent data or numbers. If a source is unreachable or covers < 70% of a month, set its features to NaN with an `avail_<source>=False` flag for that month, log it in `REPORT.md`, and continue. Every phase must run with any optional source missing.
4. No look-ahead. Every stored feature row carries `available_at = valid_time + latency` (latencies in Phase 2). Features for origin t are as-of joins on `available_at <= t`. Add a test that fails if any feature used at an origin has `available_at > origin`.
5. KMIA 1-minute data is the training **target** only. It arrives roughly 1–2 days late, so no live feature or filter observation may come from it. Training and backtests feed the filter and features from the same streams a live run sees (Phase 2, "Live" column).
6. If a Done-when check still fails after three distinct fix attempts, write the failure and the attempts to `BLOCKED.md`, keep the best version, and continue with later phases that do not depend on it. Stop entirely only if KMIA 1-minute data cannot be downloaded at all or the disk budget would be exceeded.
7. All constants live in `config.yaml`; modules read them from there. No magic numbers in code.

**Environment**

- Python 3.11+, managed with `uv`. Dependencies: numpy, pandas, pyarrow, duckdb, scipy, statsmodels, scikit-learn, lightgbm, xarray, netCDF4, cfgrib, eccodes, s3fs, herbie-data, pvlib, pyproj, ruptures, requests, tenacity, pyyaml, matplotlib, pytest.
- NOAA buckets on AWS are public: use `s3fs.S3FileSystem(anon=True)`. The default build needs no credentials or API keys.
- Politeness: IEM requests one at a time, ≥ 2 s apart, chunked by station and month, retried with exponential backoff on 429/503; on HTTP 422 halve the chunk.
- Disk budget: 150 GB under `data/`. Satellite, radar and NWP files are reduced to feature rows immediately after download and the raw file deleted.
- Long downloads are resumable jobs with a manifest of completed keys; up to 8 parallel workers for S3.

**Repo layout**

```text
kmia2h/
  config.yaml  REPORT.md  DECISIONS.md  BLOCKED.md (only if needed)
  kmia2h/
    config.py  target.py  window.py  features.py  simulate.py  run.py
    data/      stations.py fetch_iem_1min.py fetch_iem_metar.py fetch_cli.py
               fetch_ndbc.py fetch_goes.py fetch_mrms.py fetch_nwp.py
    floor/     noise_floor.py
    physics/   solar.py cloud_advect.py upstream.py rain.py ode.py batch_fit.py
    filter/    enkf.py process_model.py obs_operators.py
    residual/  residual_fit.py residual_sim.py
    mos/       train.py predict.py conformal.py
    backtest/  backtest.py scores.py attribution.py
  data/store/  Parquet tables
  params/  models/  out/phaseN/  tests/
```

**Default periods and sizes** (write these into `config.yaml`)

| Key | Default |
| --- | --- |
| Station history (KMIA, neighbours, METAR/HFMETAR, 1-minute) | 2016-01-01 → latest available |
| Satellite, radar and NWP history | 2022-01-01 → latest available |
| Noise floor and residual-process fit | 2016-01-01 → 2023-12-31 |
| Physics batch fit and MOS training | 2022-01-01 → 2023-12-31 |
| Validation (filter tuning, conformal, blending, abstain rule) | 2024-01-01 → 2024-12-31 |
| Test A | 2025-01-01 → 2025-12-31 |
| Test B (stability check) | 2026-01-01 → latest available |
| Origin cadence: noise floor / training and backtest | 10 min / 30 min |
| Paths per run: backtest / live | 2,000 / 4,000 |
| EnKF members | 300 |

The test periods are never used for fitting, tuning or feature selection.

## Overview

One run at time t ingests everything observed up to t and returns a joint probability distribution over the rounded high and low of KMIA's 1-minute temperature between t and t+2h. The model never needs a second run; re-running later answers a different, shorter-window question.

The pipeline has five layers, built in this order because each one is measured against the one before it:

1. **Target + data** — pin down exactly what settles and pull every source into one minute-resolution store (Phases 1–2).
2. **Noise floor** — measure from KMIA history how much of the 2-hour high/low is unpredictable by any model. This sets the bar every later layer is judged against (Phase 3).
3. **Mean path + state** — an energy-balance ODE whose forcing (sun, cloud, upstream air, rain) is estimated by a filter from the latest observations (Phases 4–5).
4. **Residual + extremes** — a regime-switching noise process on top of the mean path, simulated at 1-minute resolution through the exact settlement rounding to get bin probabilities (Phases 6–7).
5. **Calibration + scoring** — a statistical correction trained on years of KMIA runs, conformal adjustment, and backtests stratified by regime that tell you when the output is sharp enough to act on (Phases 8–9).

Everything is fitted on KMIA (Miami International Airport ASOS) data only. Neighbouring stations and gridded products are inputs, never training targets.

Environment, repo layout, default periods and the rules for building without supervision are in the section above.

## Phase 1 — Define the target exactly

The settled value is the highest and lowest KMIA **reported temperature** in the window, where the reported temperature each minute is the rolling 5-minute average of whole-°F one-minute values, rounded to whole °F. Every later phase calls the one function defined here.

**How ASOS produces the number** ([NWS Phoenix](https://www.weather.gov/psr/HiResASOS), [NWS Los Angeles](https://www.weather.gov/lox/asostemperature)):

- The sensor is sampled every few seconds; each minute's average is stored in whole °F.
- Each minute ASOS averages the last five 1-minute values (at least four must be valid). That rolling average, rounded, is the reported temperature.
- Official daily highs and lows are the highest and lowest rolling averages over the local-standard-time day: 00:00–23:59 EST, i.e. 05:00–04:59 UTC, all year.
- Routine METARs carry the reported temperature in tenths °C (the T group); the 5-minute HFMETAR stream carries it only in whole °C.
- An average of five whole numbers is a multiple of 0.2 °F, so exact .5 ties occur only when just four values are valid.

**Steps**

1. In `target.py` implement, vectorised over a leading paths axis:

   ```python
   def reported(t1: np.ndarray, rule: str) -> np.ndarray:
       """t1: (..., n) one-minute °F, NaN = missing. Rolling mean of the last 5 values
       (NaN if < 4 valid), rounded by `rule`. Output aligned to the last minute."""
   
   def settle(t1: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
       """t1: (..., 124) = minutes t-3 .. t+120 for origin t.
       Returns (high, low) = max / min of reported() over minutes t+1 .. t+120."""
   ```
2. Download the official climate reports: `https://mesonet.agron.iastate.edu/json/cli.py?station=KMIA&year=YYYY` for 2016 onward, into `store/cli.parquet` (date, high, low, final/preliminary flag).
3. Confirm the rounding rule. Candidates: `half_up` (floor(x + 0.5)), `half_even`, `truncate`, and `via_c` (average → °C to 0.1 → back to °F → half\_up). For each, compute the daily max and min over LST days from KMIA 1-minute data for 2020–2025 and compare with the CLI values. Keep the rule with the highest match rate and write it to `config.yaml` as `target.rounding`.
4. Also compare against the max/min of raw 1-minute values (no averaging). The NWS pages say rolling averages; keep whichever matches more days and log the comparison in `DECISIONS.md`.
5. Missing data: interpolate gaps of ≤ 3 minutes; exclude from training and scoring any window with > 5 missing minutes or a longer gap.
6. Edge statistic, used later as an abstain signal: `edge = |frac(x) − 0.5|`, where x is the unrounded max (or min) of the rolling 5-minute average of the forecast mean path.

**Done when**

- [ ] The chosen rule matches the CLI daily max on ≥ 97% and min on ≥ 97% of final-report days that have ≥ 95% 1-minute coverage. Mismatches are in `out/phase1/mismatches.csv` with a cause column (gap, sensor flag, unexplained). If the best rule is below 97%, note it in `BLOCKED.md` and continue with it.
- [ ] Unit tests pass: constant 80 → (80, 80); `[80, 80, 80, 81, 81]` → 80.4 → 80; a four-valid tie case under the chosen rule; windows with too many NaNs; vectorised and scalar results agree.

## Phase 2 — Data acquisition

Every source goes into a Parquet table keyed on UTC time, each row stamped with `available_at`. KMIA 1-minute data is only the training target; everything a live run uses comes from sources marked Live.

| # | Source | Use | Live | Latency for `available_at` | Access | Required |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | KMIA 1-minute (NCEI, via IEM) | Target, noise floor, residual fit | No — arrives \~1–2 days late | never a feature | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?station=KMIA&vars=tmpf,dwpf,sknt,drct,precip&sts=…Z&ets=…Z&sample=1min&what=download&tz=UTC` — call `?help` first to confirm station-id format and variable names | Yes |
| 2 | KMIA METAR, routine + specials | Hourly T in tenths °C (T group of the raw `metar` field), sky layers, weather codes, wind | Yes | 5 min | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=MIA&data=all&report_type=3,4&sts=…&ets=…&tz=UTC` | Yes |
| 3 | KMIA HFMETAR, 5-minute | Main live temperature stream for the filter (whole °C) | Yes | 8 min | same endpoint with `report_type=1` | Yes; synthesise gaps (step 3) |
| 4 | Neighbour ASOS: KOPF, KFLL, KHWO, KTMB, KHST, KFXE, KPMP | Upstream air temperature and wind | Yes | 8 min | same endpoint, `report_type=1,3,4` | Yes; drop a station below 70% coverage |
| 5 | NDBC FWYF1, VAKF1 | Marine air temperature and SST | Yes | 20 min | history `https://www.ndbc.noaa.gov/data/historical/stdmet/{id}h{year}.txt.gz`; live `https://www.ndbc.noaa.gov/data/realtime2/{ID}.txt` | Optional |
| 6 | GOES-East ABI L2 CONUS: ACMC (clear-sky mask), ACHAC (cloud-top height), CODC (cloud optical depth, daytime) | Cloud field and motion | Yes | 10 min | `s3://noaa-goes16/` before 2025-04-07 15:10 UTC, `s3://noaa-goes19/` after; keys `ABI-L2-ACMC/YYYY/DDD/HH/` | Optional, strongly recommended |
| 7 | MRMS composite reflectivity | Rain cells and motion | Yes | 5 min | `s3://noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/YYYYMMDD/` — list the bucket to confirm the product name | Optional |
| 8 | HRRR subhourly (RAP as fallback) | NWP trend: 2 m T, cloud, shortwave, 10 m wind | Yes | cycle + 90 min | `herbie` (model `hrrr`, product `subh`); bucket `s3://noaa-hrrr-bdp-pds` | Optional, strongly recommended |
| 9 | Solar geometry | Clear-sky irradiance, zenith angle | Yes | 0 | `pvlib`, computed | Yes |

Skipped by default: GLM lightning (adds little beyond MRMS) and the FAWN/SFWMD mesonets (no stable bulk API). Add them only if Phase 9 attribution shows offshore-flow errors dominate.

**Steps**

1. Station metadata from `https://mesonet.agron.iastate.edu/api/1/network/FL_ASOS.json` → lat, lon, elevation for KMIA and the neighbours into `params/stations.json`. Never hard-code coordinates.
2. KMIA 1-minute → `store/kmia_1min.parquet` (ts\_utc, tmpf, dwpf, sknt, drct, precip, qc) on a complete minute grid, NaN where missing. Keep values as delivered (whole °F).
3. METAR and HFMETAR for KMIA and neighbours → `store/obs.parquet` (station, ts\_utc, report\_type, tmpf, tmpc\_tenths, dwpf, sknt, drct, skyc1–4, skyl1–4, wxcodes, p01i, synthetic, available\_at). Parse tenths with the regex `T([01])(\d{3})([01])(\d{3})` (sign digit 1 = negative). For KMIA months where HFMETAR coverage is below 90%, synthesise it from 1-minute data the way ASOS does: rolling 5-minute average at minutes ending in 0 and 5 → round to °F by `target.rounding` → convert to whole °C → back to °F; set `synthetic=True`.
4. QC flags, never deletions, on KMIA 1-minute: |ΔT| > 4 °F per minute with no rain; identical values for > 45 min in daylight; values < 30 °F or > 105 °F.
5. NDBC → `store/ndbc.parquet` (station, ts\_utc, atmp\_f, wtmp\_f, wspd, wdir, available\_at).
6. GOES → `store/goes_feat.parquet`, one row per scan at 10-minute cadence for history, every scan live. Reduce each file at download, then delete it:
   - Precompute once per satellite the ABI fixed-grid pixel indices (pyproj geostationary projection from the file's `goes_imager_projection`) for a 5 km disc over KMIA and for 12 sectors of 30° × rings at 0–10, 10–20, 20–40 and 40–80 km.
   - Per region: cloud fraction (ACMC), median cloud-top height (ACHAC), median optical depth (CODC, NaN at night).
   - Motion vector (u, v in m/s) by phase cross-correlation of consecutive ACMC masks over a 200 km box, with the correlation peak stored as a quality score.
7. MRMS → `store/mrms_feat.parquet` at 10-minute cadence for history: per sector and ring the max dBZ, mean dBZ and fraction ≥ 35 dBZ, plus a motion vector computed as for GOES. Delete raw files.
8. HRRR → `store/nwp_point.parquet` keyed (cycle\_utc, valid\_utc): at the KMIA grid point and the 3 × 3 mean, `TMP:2 m`, `TCDC:entire atmosphere`, `DSWRF:surface`, `UGRD` and `VGRD` at 10 m, leads 0–6 h at 15-minute steps, `available_at = cycle + 90 min`. If a cycle is missing, the previous cycle stands in at longer lead.
9. `features.py`: `features_at(origins)` builds one row per origin by as-of joins on `available_at <= origin`. Columns include at least: last KMIA reported T (from HFMETAR/METAR) and its age in minutes; 30- and 60-minute trends; dewpoint and dewpoint depression; wind speed and direction; `sector` (onshore 30–150°, offshore 210–330°, other); `sky_class` (dominant of skyc1–4: CLR/FEW → clear, SCT → sct, BKN → bkn, OVC/VV → ovc); `rain_recent` (RA, TS or SH in wxcodes, or p01i > 0, within 30 min); neighbour and buoy temperatures; GOES and MRMS ring features; HRRR values for origin + 0…120 min; solar elevation now and at window end; `is_day`; `season` (wet May–Oct, dry Nov–Apr); hour and day-of-year as sin/cos; an `avail_*` flag per source. Precompute for the 10- and 30-minute origin grids into `store/features_10m.parquet` and `store/features_30m.parquet`.

**Done when**

- [ ] `out/phase2/coverage.csv` lists monthly coverage per source and station. KMIA 1-minute coverage is ≥ 90% in at least 90% of months since 2016; if not, note it in `BLOCKED.md` and continue.
- [ ] The HFMETAR synthesiser matches real HFMETAR exactly in ≥ 95% of reports in months that have both.
- [ ] The look-ahead test passes on 1,000 random origins.
- [ ] Unit tests pass for T-group parsing (`T02560178` → 25.6 / 17.8 °C; `T10561022` → −5.6 / −2.2 °C), the sky-class mapping, the sector rule, and ring geometry (a synthetic pixel field returns exact fractions).
- [ ] Disk use is under budget and no raw satellite or radar files remain.

## Phase 3 — Irreducible noise floor from KMIA history

The floor is the best score possible for a forecaster that knows the smooth temperature path exactly but not the minute-scale wiggle, the whole-°F quantisation, or the rounding. It is measured directly from the KMIA 1-minute record. Every later phase reports its score next to the floor's, and the gap between the two is the only headroom that exists.

**Definitions.** For origin t, with t1 the 1-minute series in whole °F:

```latex
m = \text{centred rolling mean}_L(t_1), \quad r = \text{rolling mean}_5(t_1), \quad r_m = \text{rolling mean}_5(m)
```

```latex
H_r = \max_{t<s\le t+120} r(s),\quad H_m = \max_{t<s\le t+120} r_m(s),\quad e_H = H_r - H_m\quad(\text{same for } L_r, L_m, e_L \text{ with min})
```

The realised settled high is `round(H_r)` under the Phase 1 rule. The **oracle** knows H\_m and the empirical distribution of e\_H in the current regime. It assigns each bin k the probability P(k − 0.5 ≤ H\_m + e\_H < k + 0.5). Its log-loss and hit rate are the floor.

**Procedure**

1. Load `store/kmia_1min.parquet` (tmpf, precip). Interpolate gaps of ≤ 3 minutes; longer gaps stay NaN, so any window that touches them drops out.
2. Compute m with L = 21 min (primary) and repeat everything with L = 11 and 41 for sensitivity.
3. On the 10-minute origin grid, compute H\_r, L\_r, H\_m, L\_m, e\_H, e\_L, the realised bins, `edge_H` and `edge_L`, and `rain_win` (any 1-minute precipitation in the window). Drop windows with > 5 missing minutes.
4. Attach regimes from `store/features_10m.parquet` at the origin: `season`, `is_day`, `sky_class`, wind bucket (< 5, 5–10, 10–15, > 15 kt), `sector`, dewpoint-depression bucket (< 3, 3–8, 8–15, > 15 °F), local hour.
5. Regime hierarchy for every per-regime statistic: use the finest level with ≥ 300 training windows among `[season, is_day, sky_class, wind]` → `[season, is_day, sky_class]` → `[season, is_day]`.
6. Fit on 2016–2023 and evaluate on the validation year 2024. Per regime, store the e\_H and e\_L quantiles at 1% steps, the median, the std, and Gumbel fits to e\_H and −e\_L.
7. Compute the oracle's bin probabilities on 2024, then its log-loss, mode hit rate, and hit rate by `edge` bin (0–0.1, 0.1–0.2, 0.2–0.3, 0.3–0.5). Also report the naive oracle `round(H_m)`.
8. Baseline: a point forecast from live-available data only, with HFMETAR/METAR last value T0 and 30-minute slope b: high = round(max(T0, T0 + 60b)), low = round(min(T0, T0 + 60b)). Give it a distribution the same way as the oracle (empirical residuals of H\_r about the point, per regime, fitted on 2016–2023) and score it on 2024.
9. Residual time structure, for Phase 6. On X = t1 − m, compute the autocovariance to 90 lags per regime (within days, no crossing of gaps). Fit, for lags ≥ 1, `C(k) = A_f·exp(−θ_f k) + A_s·exp(−θ_s k)`. The lag-0 excess over A\_f + A\_s is white noise, which includes the 1/12 °F² from whole-°F quantisation. Store A\_f, θ\_f, A\_s, θ\_s and the white-noise variance.

**Code**

```python
"""kmia2h/floor/noise_floor.py"""
import numpy as np, pandas as pd
from scipy import stats, optimize
from kmia2h.config import CFG
from kmia2h.target import round_rule

H, RULE = 120, CFG.target.rounding
LEVELS = [['season', 'is_day', 'sky_class', 'wind'], ['season', 'is_day', 'sky_class'], ['season', 'is_day']]

def fwd(x: pd.Series, n: int, how: str) -> pd.Series:
    """Extreme of x over minutes t+1 .. t+n, aligned at t; NaN if any value is NaN."""
    r = x.iloc[::-1].rolling(n, min_periods=n)
    return (r.max() if how == 'max' else r.min()).iloc[::-1].shift(-1)

def fill_short(x: pd.Series, k: int = 3) -> pd.Series:
    na = x.isna()
    run = na.groupby((~na).cumsum()).transform('sum')
    return x.interpolate(limit_area='inside').where(~(na & (run > k)))

def floor_table(t1: pd.Series, precip: pd.Series, L: int, step_min: int = 10) -> pd.DataFrame:
    t1f = fill_short(t1)
    m = t1f.rolling(L, center=True, min_periods=L).mean()
    r, rm = t1f.rolling(5, min_periods=4).mean(), m.rolling(5, min_periods=5).mean()
    nmiss = t1.isna().astype(float).iloc[::-1].rolling(H).sum().iloc[::-1].shift(-1)
    d = pd.DataFrame({'Hr': fwd(r, H, 'max'), 'Lr': fwd(r, H, 'min'),
                      'Hm': fwd(rm, H, 'max'), 'Lm': fwd(rm, H, 'min'), 'nmiss': nmiss,
                      'rain_win': fwd((precip.fillna(0) > 0).astype(float), H, 'max') > 0})
    d = d[(d.index.minute % step_min == 0) & (d.nmiss <= 5)].dropna()
    d['eH'], d['eL'] = d.Hr - d.Hm, d.Lr - d.Lm
    d['hi'], d['lo'] = round_rule(d.Hr, RULE), round_rule(d.Lr, RULE)
    d['edge_H'], d['edge_L'] = (d.Hm % 1 - 0.5).abs(), (d.Lm % 1 - 0.5).abs()
    return d

def assign_regime(d: pd.DataFrame, train: pd.Series, min_n: int = 300) -> pd.Series:
    """Finest regime key with >= min_n training windows."""
    key = pd.Series(None, index=d.index, dtype=object)
    for lv in LEVELS:
        k = d[lv].astype(str).agg('|'.join, axis=1)
        ok = k.map(k[train].value_counts()).fillna(0) >= min_n
        key = key.where(key.notna() | ~ok, k)
    return key

def bin_probs(point: np.ndarray, resid_sorted: np.ndarray, offsets=range(-15, 16)):
    """P(round(point + e) = k) for k = floor(point) + offset, e ~ empirical residuals."""
    base = np.floor(point)
    ks = base[:, None] + np.array(list(offsets))[None, :]
    cdf = lambda x: np.searchsorted(resid_sorted, x, side='left') / len(resid_sorted)
    p = cdf(ks + 0.5 - point[:, None]) - cdf(ks - 0.5 - point[:, None])
    return ks, np.clip(p, 1e-4, None) / np.clip(p, 1e-4, None).sum(1, keepdims=True)

def score(d, point_col, resid_col, realised_col, key, train, test):
    out = []
    for g, rows in d[test].groupby(key[test]):
        res = np.sort(d.loc[train & (key == g), resid_col].values)
        ks, p = bin_probs(rows[point_col].values, res)
        hit = ks == rows[realised_col].values[:, None]
        out.append(pd.DataFrame({'logloss': -np.log(np.maximum((p * hit).sum(1), 1e-4)),
                                 'mode_hit': ks[np.arange(len(ks)), p.argmax(1)] == rows[realised_col].values},
                                index=rows.index))
    return pd.concat(out)

def acov_fit(x: np.ndarray, maxlag: int = 90) -> dict:
    x = x - np.nanmean(x)
    c = np.array([np.nanmean(x[:len(x) - k] * x[k:]) for k in range(maxlag + 1)])
    k = np.arange(1, maxlag + 1)
    f = lambda k, af, tf, as_, ts: af * np.exp(-tf * k) + as_ * np.exp(-ts * k)
    (af, tf, as_, ts), _ = optimize.curve_fit(
        f, k, c[1:], p0=[c[1] * .6, .3, c[1] * .4, .04], bounds=(0, np.inf), maxfev=20000)
    if tf < ts: af, tf, as_, ts = as_, ts, af, tf
    return dict(A_f=af, theta_f=tf, A_s=as_, theta_s=ts, white=max(c[0] - af - as_, 0.0))
```

The agent fills in the driver (`main`) that loops over L ∈ {11, 21, 41}, attaches regimes, runs `score()` for the oracle (`point='Hm', resid='eH', realised='hi'`, and likewise for the low), the naive oracle and the baseline, and writes the outputs below.

**Outputs**

- `out/phase3/summary.md`: for each L, rows {all, wet-day, wet-night, dry-day, dry-night} × columns {n, std e\_H, std e\_L, naive-oracle hit, oracle hit, oracle log-loss, baseline hit, baseline log-loss}, for high and low, plus hit rate by `edge` bin.
- `out/phase3/regime_*.csv`: full per-regime tables.
- `params/floor.json`: per regime the e\_H and e\_L quantiles, medians, std and Gumbel parameters.
- `params/residual_acov.json`: per regime A\_f, θ\_f, A\_s, θ\_s and white-noise variance.

**Done when**

- [ ] `floor_table` runs over 2016–2023 in under 10 minutes.
- [ ] A synthetic test passes: a smooth sinusoid with no noise gives |e\_H| < 0.05 and 100% oracle hit; adding white noise with σ = 0.5 °F makes the mean e\_H clearly positive.
- [ ] On 2024, the oracle's log-loss is lower than the baseline's for both high and low. If not, there is a bug: fix it before moving on.
- [ ] All output files exist, and `REPORT.md` states the floor and baseline hit rates and log-losses for L = 21.

## Phase 4 — Mean-path physics

The mean path m(s) over the next 120 minutes comes from a surface energy-balance ODE whose forcing terms are built from the data at run time. This replaces the fixed long-run mean in a plain OU model with a mean that moves with sun, cloud, upstream air and rain.

```latex
\frac{dT}{dt} = a\,S_{clr}(t)\,\big(1-\kappa\,C(t)\big) \;-\; b\,\big(T - T_{adv}(t)\big) \;-\; c\,\sigma_{SB}\big(T^4 - T_{sky}^4\big)\,(1-\lambda C(t)) \;-\; R(t)
```

Terms, left to right: solar heating attenuated by cloud; relaxation toward the advected upstream air temperature; net longwave cooling (dominant at night, suppressed by cloud); rain/outflow cooling. Coefficients a, b, c, κ, λ are states estimated by the filter in Phase 5, not constants — but you need initial values and a climatological prior from a batch fit first.

**Step 1 — Clear-sky solar S\_clr(t).** Use `pvlib` with KMIA's latitude, longitude and elevation from `params/stations.json`, the Ineichen clear-sky model, and monthly Linke turbidity from `pvlib.clearsky.lookup_linke_turbidity`. Output GHI in W m⁻², solar zenith and `is_day` (elevation > 0) at 1-minute steps.

**Step 2 — Effective cloud factor C(t) for the next 2 h.** This is the term with the most skill in the daytime and the one you must build from satellite, not from the station.

1. Take cloud fraction over KMIA (5 km disc) and in each upwind sector and ring from `store/goes_feat.parquet` (Phase 2).
2. Advection vector: the GOES cloud-motion vector when its quality score is above a threshold tuned on 2024; otherwise the HRRR 10 m wind at origin times a factor fitted in Step 5 (default 1.5, since low clouds move faster than the surface wind).
3. Advect: C(t+Δ) is the cloud fraction now at distance |v|·Δ upwind, interpolated between rings. Beyond 80 km, relax toward the HRRR total-cloud forecast for that valid time.
4. Convective growth: in the wet season between 10:00 and 16:00 local, add the HRRR total-cloud change between origin and valid time.
5. Optical thickness: κ per cloud class from CODC and ACHAC (thin or high: optical depth < 3 or top > 8 km; otherwise thick), fitted in Step 5. At night, use cloud-top height only.
6. Fallback when GOES is unavailable: C from the latest METAR sky cover (CLR 0, FEW 0.2, SCT 0.4, BKN 0.75, OVC/VV 1.0), held constant, with κ at the thick value.

**Step 3 — Upstream advected temperature T\_adv(t).**

1. Upwind set by the HRRR 10 m wind direction at origin: onshore (30°–150°) → VAKF1, FWYF1, falling back to KFLL; offshore (210°–330°) → KTMB, KHST; northerly (330°–30°) → KOPF, KHWO, KFXE, KFLL, KPMP; anything else → all neighbours. Use HRRR wind, not KMIA's own, which the airport surface distorts.
2. Lag each station by its distance to KMIA divided by the wind speed, capped at 120 min.
3. T\_adv = inverse-distance-weighted mean of the lagged upstream temperatures; missing stations drop out of the weights. If none is available, T\_adv = HRRR 2 m T at KMIA.
4. Sea-breeze switch: onset when KMIA wind turns into 30°–150° with a dewpoint rise ≥ 2 °F within 30 min, or when the HRRR 10 m wind at KMIA turns onshore inside the window. After onset, T\_adv takes the marine value and b switches to `b_onshore` (fitted in Step 5). This caps the afternoon high in Miami and is the most common warm-season cause of a one-bin miss.

**Step 4 — Rain/outflow cooling R(t).**

1. From `store/mrms_feat.parquet`, find upwind sector-rings within 40 km with max reflectivity ≥ 35 dBZ, and take the MRMS motion vector.
2. If such a cell is projected to pass within 5 km of KMIA inside the window, R(t) is a pulse: onset at projected arrival, drop size from a regression fitted on 2022–2023 of observed KMIA drops on max dBZ and the fraction ≥ 35 dBZ, then exponential recovery (time constant fitted, default 40 min).
3. Otherwise R(t) = 0. The chance of a new cell forming belongs to the residual's jump process in Phase 6, not the mean path.

**Step 5 — Batch-fit the climatological coefficients.** Before any filtering, fit a, b, c, κ, λ by nonlinear least squares over 2 years of KMIA data, integrating the ODE in 1-minute steps from the observed T at each hour and matching the following 120 minutes. Fit separately for day/night and for the three wind sectors. These become the filter's prior means and the bounds on its state.

**Step 6 — Integrate.** With forcing series and coefficients in hand, `scipy.integrate.solve_ivp` (RK45) from T(t) over 120 minutes gives m(s). Store it at 1-minute resolution along with each forcing series so Phase 9 can attribute errors to the term that caused them.

Deliverables: `solar.py`, `cloud_advect.py`, `upstream.py`, `rain.py`, `ode.py`, `batch_fit.py`, and a `params/ode_prior.json` with coefficients by day/night × sector.

**Done when**

- [ ] The batch fit converges for every day/night × sector group, all coefficients sit inside physical bounds (a, b, c, κ ≥ 0, κ ≤ 1), and they are saved to `params/ode_prior.json`.
- [ ] On 2024, the deterministic mean path (no filter, no noise) beats the persistence-plus-trend baseline in RMSE over the 120-minute path and in RMSE of H\_r and L\_r. Both are in `REPORT.md`.
- [ ] `out/phase4/ablation.csv` gives the RMSE of H\_r and L\_r with each forcing term removed in turn (cloud, advection, rain, longwave), by season × day/night.
- [ ] Unit tests pass: with zero forcing the ODE relaxes exponentially to T\_adv; the solar term is zero at night; the advection lag is right for a synthetic wind field.

## Phase 5 — State estimation (ensemble Kalman filter)

The 2-hour forecast error is dominated by how well you know the state at run time, so the filter matters more than the ODE's sophistication. It runs continuously on the 1-minute stream and, at run time t, hands the simulator a posterior ensemble rather than a single starting point.

**State vector** (per ensemble member):

```latex
x = \big[\,T,\;\dot T,\;T_{-1},\;T_{-2},\;T_{-3},\;C_{eff},\;a,\;b,\;c,\;\kappa,\;X_{slow},\;X_{fast}\,\big]
```

T and its tendency are the physical state. T₋₁ to T₋₃ are the three previous 1-minute temperatures, carried so that `settle()` gets the four lead minutes before the window and so that observations of 5-minute averages can be assimilated. C\_eff is the cloud factor the sensor actually feels (satellite C is an observation of it). a, b, c and κ are ODE coefficients allowed to drift slowly. X\_slow and X\_fast are the two residual components from Phase 6, so the filter also knows where the noise currently sits.

**Why ensemble, not linear Kalman.** The ODE is nonlinear in T (longwave term) and in the coefficient × forcing products, and the residual is regime-switching. An EnKF with 200–500 members handles both without linearisation and gives the simulator a ready-made ensemble. `filterpy` has an `EnsembleKalmanFilter`; writing your own is \~60 lines and lets you control inflation.

**Observations** — only streams a live run has (Phase 2, Live column), each assimilated at its valid time once its `available_at` has passed:

- KMIA HFMETAR, every 5 min, whole °C. Observation operator: the 5-minute mean of the member's last five 1-minute values → round to °F → convert and round to whole °C → back to °F. Model the rounding as uniform error, σ ≈ 0.52 °F (1.8 °F / √12).
- KMIA routine METAR, hourly, T group in tenths °C. Same 5-minute-mean operator, σ ≈ 0.3 °F.
- GOES cloud fraction over the station, σ ≈ 0.15, as an observation of C\_eff (daytime and night).
- HRRR 2 m T at the matched valid time, σ ≈ 1.5 °F — weak but it anchors the trend.
- T\_adv from the upstream stations, with σ from the Phase 3 residual statistics.

Observations arrive 5–10 minutes after their valid time. Keep each member's state for the last 30 minutes; when an observation arrives, update the members at its valid time and re-propagate them to now.

**Steps.**

1. **Process model.** Advance each member one minute: integrate the ODE with that member's coefficients and forcing, shift the lag states, add AR(1) innovations to X\_slow and X\_fast with the Phase 6 regime parameters, and random-walk the coefficients with small variance so they adapt over hours, not minutes.
2. **Update.** EnKF analysis with perturbed observations at each observation's valid time, then re-propagate, as described above.
3. **Inflation.** Multiplicative, between 1.02 and 1.05, tuned on 2024.
4. **Constraints.** Clip a, b, c and κ to the physical bounds from the Phase 4 batch fit; keep C\_eff in \[0, 1\].
5. **Regimes.** Each minute, classify the regime (day/night × sector × sky) and switch the residual parameters. Never re-initialise the ensemble on a switch.
6. **Spin-up.** Start 6 hours before the first origin needed.
7. **Output at origin t.** The analysis ensemble `{x_i(t)}`, i = 1…300, including the lag states, plus each member's smoothed path over the last 120 minutes (needed for elapsed windows in Phase 7).
8. **Backtests.** Run the filter continuously through each period on the live streams (real HFMETAR where it exists, synthetic where Phase 2 had to fill it) and snapshot the ensemble at every 30-minute origin to `store/enkf_snapshots/`. KMIA 1-minute data is used only to score.

**Validation.** Over a held-out year, plot the filter's one-minute-ahead innovation (obs − prediction): it should be white, mean zero, and its variance should match the predicted variance (rank histogram flat). If innovations are autocorrelated, the residual model in Phase 6 is missing a timescale; if the rank histogram is U-shaped, increase inflation.

Deliverables: `enkf.py`, `process_model.py`, a tuned `params/filter.json` (N, inflation, obs errors), and the innovation diagnostics notebook.

**Done when**

- [ ] On 2024, normalised HFMETAR innovations have mean within ±0.1, variance between 0.8 and 1.2, and lag-1 autocorrelation < 0.2.
- [ ] The rank of the true 1-minute temperature at each origin within the analysis ensemble is roughly uniform: χ² test p > 0.01 by season × day/night, or the deviation is logged in `REPORT.md`.
- [ ] The analysis RMSE of T at origin, against 1-minute truth, is lower than that of the last HFMETAR value alone.
- [ ] Filtering one full year takes under 2 hours.

## Phase 6 — Residual process

The residual X(s) = T(s) − m(s) is modelled as two timescales plus a daytime regime switch, with parameters fitted per regime from the Phase 3 decomposition. This is the layer that turns the mean path into a distribution and gives the high its positive skew and the afternoon low its fat negative tail.

```latex
X(s) = X_{slow}(s) + X_{fast}(s)
```

```latex
dX_{fast} = -\theta_f X_{fast}\,ds + \sigma_f\, dW_f, \qquad \theta_f^{-1} \approx 2\text{–}5\ \text{min}
```

```latex
dX_{slow} = -\theta_s X_{slow}\,ds + \sigma_s\, dW_s + J\,dN_s, \qquad \theta_s^{-1} \approx 15\text{–}40\ \text{min}
```

The jump term J dN\_s is active only in daytime with SCT/BKN cloud: N\_s is a Poisson process whose rate is the cloud-edge passage frequency, and J is the sun/shade temperature step. At night both are zero and X\_slow is plain OU.

**Fitting steps, all on KMIA history, per regime (day/night × wind × sky × season).**

1. **Separate timescales.** From the Phase 3 residual X, compute the ACF to 90 lags per regime. Fit a sum of two exponentials `ρ(k) = w e^{−θ_f k} + (1−w) e^{−θ_s k}`. If the second term's weight is < 0.1, use a single OU for that regime.
2. **Variances.** σ\_f² and σ\_s² from the fitted weights and the regime's total residual variance.
3. **Jump detection.** In daytime regimes, run a change-point detector (PELT on the 5-minute mean of X, or a simple |ΔX\_5min| > 1.0 °F threshold). Count jumps per hour → Poisson rate λ by sky class and hour; record jump sizes → fit a two-sided distribution (drops under cloud, rises when it clears). Rates will be highest for SCT/BKN cumulus in the 11:00–16:00 local window.
4. **Heavy tails.** Compare the empirical kurtosis of X\_fast innovations to Gaussian. If excess kurtosis > 1, use Student-t innovations with fitted ν rather than Gaussian; this matters for the 99th percentile of e\_H.
5. **Validate against the floor.** Simulate the fitted residual process over 120-minute windows (no mean-path uncertainty) and recompute e\_H and e\_L. Their distributions should match the Phase 3 Gumbel fits per regime within sampling error. If they don't, the residual model is wrong and nothing downstream can fix it — iterate here.
6. **Store** per-regime parameters in `params/residual.json`: θ\_f, σ\_f, θ\_s, σ\_s, λ, jump-size distribution, ν.

**Parsimony check.** A two-timescale, regime-switching OU with jumps has \~8 parameters per regime and perhaps 30 regimes. With 5 years of 1-minute data that is well determined. Resist adding more structure until Phase 9 shows a specific, regime-localised miscalibration.

Deliverables: `residual_fit.py`, `residual_sim.py`, `params/residual.json`, and the validation plot of simulated vs empirical e\_H / e\_L per regime.

**Implementation notes**

- Start the fit from `params/residual_acov.json` (Phase 3) rather than re-deriving the ACF.
- The white-noise variance from Phase 3 includes 1/12 °F² of whole-°F quantisation. Do not simulate that part as noise. Instead, add the remaining white noise to each simulated 1-minute value, then **round every simulated 1-minute value to whole °F**, exactly like the real sensor, before `settle()` sees it.

**Done when**

- [ ] With the true smooth path as the mean, the simulated e\_H and e\_L match the Phase 3 quantiles per regime (two-sample KS statistic < 0.05, or the misfit is logged per regime).
- [ ] Unit tests pass: the simulated OU stationary variance matches theory within 5%; a jump rate is recovered from simulated data within 10%; simulated 1-minute values are whole numbers.

## Phase 7 — Monte Carlo through the exact settlement operator

The forecast is produced by simulating N full 1-minute paths from the filter's analysis ensemble, applying `settle()` to each, and counting. No closed-form extreme-value approximation is used in the output; the Gumbel fits from Phase 3 serve only as a check.

**Steps.**

1. **Seed.** Take the 300 analysis members from Phase 5, including their lag states T₋₃…T₀. Resample with replacement to N paths (2,000 in backtests, 4,000 live) and jitter T by the analysis spread.
2. **Forcing ensemble.** Each path gets its own forcing: cloud-advection vector ± its estimated uncertainty, cloud fraction ± 0.1, upstream T ± the station residual std, rain-cell arrival ± 10 min and drop size ± its regression std. Without forcing perturbation the ensemble is too narrow exactly when it matters.
3. **Integrate.** For each path, step the ODE, X\_slow, X\_fast and jumps for 120 one-minute steps (Euler–Maruyama), add the remaining white noise, and round each 1-minute value to whole °F. Vectorise over paths with numpy.
4. **Apply the settlement operator.** Prepend the four lag minutes and call `settle(paths, rule)` from Phase 1.
5. **Count.** `P(high = k) = mean(highs == k)`, the same for low, and the joint table `P(high = k, low = j)`. Also store the quantiles of the unrounded path max and min and the `edge` of the ensemble-mean path.
6. **Sanity checks every run.** The spread of `max(path) − max(mean path)` lies within the Phase 3 range for the regime; the ensemble mean at Δ = 1 min is within 2 °F of the latest observation; no path moves more than 4 °F in one minute outside a rain pulse.

**Elapsed window.** If the market window began before t, the elapsed part is not known exactly in a live run: HFMETAR gives whole °C every 5 minutes and METAR gives tenths once an hour. Join each simulated future path to a smoothed member path for the elapsed minutes (Phase 5, step 7) and apply `settle()` over the whole market window. When the observations already pin the high above anything the future paths reach, the distribution collapses to one or two bins on its own.

**Output schema** (`forecast.json` per run):

```json
{"origin_utc": "…", "window_end_utc": "…", "n_paths": 4000,
 "p_high": {"81": 0.03, "82": 0.71, "83": 0.26},
 "p_low":  {"77": 0.12, "78": 0.88},
 "p_joint": {"82|78": 0.62, "83|78": 0.24, "…": "…"},
 "q_max": {"05": 81.6, "50": 82.3, "95": 83.1},
 "q_min": {"05": 77.2, "50": 77.9, "95": 78.4},
 "edge_high": 0.31, "edge_low": 0.08,
 "regime": "day-onshore-SCT-mod", "mean_path": ["…120 values…"]}
```

Deliverables: `simulate.py`, `forecast_schema.json`, and a notebook that runs one historical origin end-to-end and plots paths, the mean path, and the bin histograms.

**Done when**

- [ ] One run with 4,000 paths takes under 5 s on a laptop CPU.
- [ ] On 2024, the raw ensemble's log-loss and hit rate for high and low are in `REPORT.md` next to the floor and the baseline, together with a rank histogram of realised H\_r within the path maxima.
- [ ] Every `forecast.json` validates against `forecast_schema.json`, and each probability table sums to 1 within 1e-9.
- [ ] Runs are reproducible: the same origin and seed give identical output.

## Phase 8 — Post-processing: MOS + conformal calibration

The physics ensemble will be systematically too narrow or biased in some regimes. A statistical layer trained on years of historical runs corrects that, and a conformal step guarantees the stated probabilities are honest. This is standard model-output-statistics practice; the only difference is that the target is an extreme over a window, not a point value.

**Steps.**

1. **Build the training set.** Run Phases 4–7 at every 30-minute origin of 2022–2023 (MOS train) and of 2024 (validation), with the filter fed only by live streams. Store each forecast with its realised `settle()` outcome from 1-minute data — about 35,000 training and 17,500 validation rows.
2. **Features.** Physics quantiles of the path max and min (5/25/50/75/95), `edge`, regime one-hots, hour and day-of-year as sin/cos, ensemble spread, last observed T and its age, 30-minute trend, upstream T anomaly, current satellite cloud fraction, max upstream reflectivity, HRRR 2 m T at window end, and the `avail_*` flags.
3. **Quantile gradient boosting.** LightGBM with pinball loss for the quantiles 0.02, 0.05, 0.10 … 0.90, 0.95, 0.98 of the unrounded realised H\_r, and separately of L\_r. Use monotone constraints on the physics-quantile features. Tune with 3-fold time-blocked CV inside 2022–2023; defaults are `num_leaves=31`, `learning_rate=0.03`, up to 2,000 rounds with early stopping.
4. **Quantiles to bins.** Sort the predicted quantiles to remove crossings, build a piecewise-linear CDF with exponential tails beyond the outer quantiles, and set `P(high = k) = F(k + 0.5) − F(k − 0.5)` under the Phase 1 rounding. Do the same for the low. For the joint table, use a Gaussian copula with the correlation of the physics ensemble's (max, min) pairs.
5. **Conformal calibration.** On 2024, apply split-conformal adjustment per season × day/night so that each central interval reaches its nominal coverage.
6. **Blend.** Final `P = w · MOS + (1 − w) · physics`, with w chosen per season × day/night on a 0–1 grid in steps of 0.05 to minimise 2024 log-loss. If w → 1, Phase 4 adds nothing; if w → 0, the MOS is overfitting. Log either case.
7. **Retraining in live use.** Retrain MOS monthly on a rolling window; refit conformal weekly. If a step change appears in the bias (sensor replacement), reset the training window and log it.

Deliverables: `mos_train.py`, `mos_predict.py`, `conformal.py`, `models/mos_high.lgb`, `models/mos_low.lgb`, `params/conformal.json`.

**Done when**

- [ ] On 2024, the blended log-loss is no worse than the raw ensemble's, for both high and low.
- [ ] Reliability on 2024: in every probability decile, |observed frequency − predicted| < 0.05, or the deviation is logged.
- [ ] No crossed quantiles in outputs, and all probability tables sum to 1.

## Phase 9 — Backtesting, scoring and abstain rules

Score the model on a test year it never saw, against the noise floor and the persistence baseline, stratified by regime and by `edge`. The result is a table that tells you in which situations the output is sharp enough to act on and in which it is not.

**Scores to compute for both high and low**

| Score | What it measures | Reference values |
| --- | --- | --- |
| Bin log-loss: −log P(k\_realised) | Primary; what a market pays for | Floor model (oracle mean path + fitted residual) and persistence baseline |
| Bin hit rate of the mode | Readable, but ignores confidence | Floor ceiling from Phase 3 |
| CRPS on the continuous max/min | Sharpness + calibration together | Persistence baseline |
| Reliability diagram per P-bucket | Is a stated 70% really 70%? | Diagonal |
| Sharpness: mean of max\_k P(k) | How often you commit | Higher is better *only if reliability holds* |
| Rank histogram of realised max in the ensemble | Spread correctness | Flat |

**Steps.**

1. **Run** the frozen pipeline (no refitting) at every 30-minute origin of Test A (2025), with live-stream inputs only. Store each forecast with its outcome.
2. **Compute** all scores globally, then by local hour, season, wind sector, sky class, `rain_win` and `edge` bin (0–0.1, 0.1–0.2, 0.2–0.3, 0.3–0.5).
3. **Compare** with the floor (the Phase 3 oracle evaluated on the same origins, using its 2016–2023 residual fits), with the baseline, and with the raw ensemble before MOS. Where the gap to the floor is large, use the stored forcing series to attribute the error to cloud advection, sea-breeze timing, rain cells or coefficients.
4. **Attribution.** For the 50 worst log-loss windows, save a PNG of observed T, the mean path, a sample of 50 ensemble paths and each forcing term, and classify each window's failure in a CSV.
5. **Abstain rule, fitted on 2024 only.** Logistic model of P(mode correct | edge, mode probability, regime, ensemble spread). Choose the threshold on 2024 so that acted-on origins hit ≥ 85% (or the best achievable, logged). Then evaluate that fixed rule on Test A.
6. **Stability.** Repeat steps 1–2 on Test B. Flag any score that differs from Test A by more than two standard errors (block bootstrap by day).

Deliverables: `backtest.py`, `scores.py`, the regime × edge score tables, the attribution notebook, and `params/abstain.json`.

**Done when**

- [ ] `out/phase9/scores.md` holds the global and stratified tables for Test A and Test B, each with floor, baseline, raw ensemble and final model.
- [ ] The final model's log-loss is lower than the baseline's on Test A for both high and low.
- [ ] For the abstain rule, the Test A hit rate among acted-on origins is reported with a bootstrap 95% interval, along with the share of origins acted on.
- [ ] The 50 attribution plots and their failure classification are in `out/phase9/worst/`.

## Phase 10 — The live run

A single command at any time t produces the 2-hour high/low distribution from everything available at t. It is one pass through the trained pipeline; nothing is refit at run time.

**Pipeline for a run at 10:39 local (15:39 UTC):**

1. **Fetch the latest of every live source.** That means KMIA HFMETAR and METARs (the newest is usually 5–10 minutes old), neighbours, buoys, the newest GOES ACMC/ACHAC/CODC scans, the newest MRMS file, and the newest complete HRRR cycle. KMIA 1-minute data does not exist yet for this period and is not used.
2. **Update the filter** from its persisted state, assimilating every observation that has become available, and propagate to the origin (15:39 UTC). If the saved state is more than 6 hours old, spin up from the last 6 hours of live streams.
3. **Build forcing** for 15:40–17:39 UTC: solar from `pvlib`, advected cloud, lagged upstream T, rain pulses from MRMS, and the HRRR trend.
4. **Simulate** 4,000 paths (Phase 7) and apply `settle()`.
5. **Post-process** with MOS, conformal adjustment and blending (Phase 8).
6. **Compose with the elapsed window** if `--window-start` is earlier than the origin.
7. **Apply the abstain rule** (Phase 9). Write `forecast.json` plus a one-line summary: `high: 82 (0.71) | low: 78 (0.88) | regime day-onshore-sct | edge_high 0.31 | ACT / ABSTAIN`.
8. **Log** the forecast. A daily scheduled job fetches the KMIA 1-minute archive once it lands, about 1–2 days later, and scores past forecasts into `out/live_scores.parquet`.

**Runtime budget.** Fetch 20–40 s (dominated by GOES/MRMS downloads), filter < 1 s, forcing 2–5 s, simulation < 2 s, MOS < 1 s. Total under a minute, so origin and run time differ by at most one observation.

**Graceful degradation.** If GOES is missing, use the last scan, let the filter carry C\_eff, and double the cloud perturbation. If MRMS is missing, set R = 0 and raise the jump rate to the regime's 90th percentile. If HRRR is missing, use the previous cycle at longer lead. If there has been no KMIA HFMETAR or METAR in the last 20 minutes, ABSTAIN. Never substitute a neighbour station for the target.

**Optional upgrade.** If the environment variable `SYNOPTIC_TOKEN` is set, also assimilate real-time 1-minute ASOS from Synoptic's HF-ASOS service (2–5 minute latency) as an extra observation stream with a 1-minute observation operator. The default build does not need it.

Deliverable: `run.py --origin now` and `run.py --origin 2026-11-08T15:39Z` (for reproducing any historical run), plus a cron/launchd entry if you want runs on a fixed cadence.

**Done when**

- [ ] `run.py --origin <a Test A timestamp>` reproduces the stored Test A forecast for that origin exactly, given the same seed.
- [ ] `run.py --origin now` finishes in under 90 seconds and writes `forecast.json`.
- [ ] Outage tests, with GOES, MRMS, HRRR and the KMIA observations removed one at a time, each produce either a forecast or ABSTAIN and never crash.

## Build order

Build in this order. Each step is testable on its own; commit and update `REPORT.md` after each.

- [ ] 1\. Phase 1: download the CLI archive, confirm the rounding and averaging rule, freeze `settle()`.
- [ ] 2\. Phase 2, sources 1–4 and 9: KMIA 1-minute, METAR and HFMETAR (with the synthesiser), neighbours, station metadata, solar; build the feature tables without satellite, radar or NWP.
- [ ] 3\. Phase 3: noise floor, oracle and baseline scores, residual autocovariance.
- [ ] 4\. Phase 2, sources 5–8: buoys, GOES, MRMS, HRRR. These downloads are long, so start them in the background and continue with steps 5–6 meanwhile.
- [ ] 5\. Phase 4 using the METAR sky-cover fallback for cloud: batch fit and ablation.
- [ ] 6\. Phases 5–7 with a single-timescale OU residual: the first end-to-end probabilistic forecast, scored on 2024 against the floor and the baseline.
- [ ] 7\. Phase 4 in full: GOES cloud advection, upstream advection with the sea-breeze switch, MRMS rain pulses. Re-score 2024 after each addition.
- [ ] 8\. Phase 6 in full: two timescales, jumps, heavy tails, quantisation. Re-validate against the floor.
- [ ] 9\. Phase 8: MOS, conformal calibration, blending.
- [ ] 10\. Phase 9: Test A and Test B, abstain rule, attribution.
- [ ] 11\. Phase 10: `run.py`, persisted filter state, outage tests, daily scoring job.

After step 6 there is a usable model with a known gap to the floor; steps 7–8 close that gap regime by regime, guided by the 2024 scores. Test A and Test B are touched only in step 10.

## Sources

- [NWS Phoenix — High-resolution ASOS data](https://www.weather.gov/psr/HiResASOS): whole-°F 1-minute values, rolling 5-minute averages, how daily highs and lows are taken.
- [NWS Los Angeles — ASOS temperature](https://www.weather.gov/lox/asostemperature): METAR tenths vs 5-minute whole-°C reports; 1-minute data not public in real time.
- [IEM asos1min.py documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?help): NCEI 1-minute archive, about 24-hour delay.
- [IEM asos.py documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help): METAR download, report types including HFMETAR.
- [IEM cli.py documentation](https://www.mesonet.agron.iastate.edu/json/cli.py?help): NWS daily climate report values.
- [NOAA OSPO — GOES-19 operational as GOES-East](https://www.ospo.noaa.gov/data/messages/2025/04/MSG_20250407_1510.html): switch on 7 April 2025, 15:10 UTC.
- [Registry of Open Data on AWS — NOAA GOES](https://registry.opendata.aws/noaa-goes/): public GOES-16 and GOES-19 buckets.
- [Synoptic — High-frequency ASOS](https://docs.synopticdata.com/services/high-frequency-asos): real-time 1-minute ASOS, 2–5 minute latency.
