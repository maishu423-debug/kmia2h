# REPORT

Rebuild of the KMIA 2-hour high/low model on a new machine (REBUILD.md), 2026-11-13.

## Phase 1 — target (build step 1)

Daily max / min of the rolling 5-minute mean over LST days, 2020-2025, final CLI reports, days with >= 95% 1-minute coverage after QC exclusion (n = 1,547).

| Rule | High match | Low match |
| --- | --- | --- |
| **half_up (chosen)** | **99.48%** | **98.13%** |
| half_even | 99.48% | 98.13% |
| via_c | 99.48% | 98.13% |
| truncate | 61.3% | 75.8% |
| raw 1-minute max/min | 31.7% | 68.2% |

First build: 99.46% / 98.07%. Same rule; the small difference is the QC exclusion set (see DECISIONS.md) and its effect on which days pass the coverage test (1,547 days here). Mismatches with causes are in `out/phase1/mismatches.csv` (high: 3 gap, 4 sensor flag, 1 unexplained; low: 10 gap, 7 sensor flag, 12 unexplained).

## Phase 2 — data, sources 1-4 and 9 (build step 2)

- KMIA 1-minute 2016-01 to 2025-12 (5.26 M minutes). QC: 1,844 range-flagged minutes (including the 122 F stretch of 2023-07-16/17), 31,783 stuck-sensor minutes, 750 isolated spikes (kept, see DECISIONS.md).
- Coverage (`out/phase2/coverage.csv`): KMIA 1-minute months >= 90%: **69.4%** (2016-2024) -> BLOCKED.md. KMIA HFMETAR mean 93.5%, routine METAR 99.9%.
- HFMETAR synthesiser: matches real HFMETAR exactly on **99.58%** of reports (alignment: 5-minute block ending at the report minute; one minute earlier gives 95.2%). Synthesised months: 2016-01..04, 2017-07, 2020-11, 2020-12.
- NDBC: VAKF1 2016-2024 (96% hourly ATMP coverage); FWYF1 ends 2022-12-18.
- Look-ahead test: features at 1,000 random origins, and a rebuild from observations truncated at 100 of them, agree exactly.

## Phase 3 — noise floor (build step 3)

L = 21 min, fit 2016-2023, scored on 2024 (n = 43,434 windows). `out/phase3/summary.md` has L = 11 / 41 and edge bins.

| | High hit | High log-loss | Low hit | Low log-loss |
| --- | --- | --- | --- | --- |
| Naive oracle round(H_m) | 0.710 | | 0.770 | |
| **Oracle (floor)** | **0.819** | **0.456** | **0.839** | **0.428** |
| **Persistence + trend baseline** | **0.288** | **2.060** | **0.260** | **2.145** |
| First build: floor | 0.821 | 0.451 | 0.836 | 0.425 |
| First build: baseline | 0.293 | 2.054 | 0.271 | 2.127 |

Floor and baseline reproduce within 0.01 in hit rate and 0.02 in log-loss. Differences come from the QC exclusion set and the windows that pass the missing-minute rule. Mean e_H = settled-high gap is in `params/floor.json` per regime.

Residual autocovariance: X = t1 - m with a centred 21-minute mean has a negative lobe at lags 5-10 and almost no structure past 15 min, so the two-exponential fit over 90 lags was degenerate (kept in `out/phase3/residual_acov_fullfit.json`). `params/residual_acov.json` holds a single-exponential fit on lags before the first zero crossing (theta ~ 0.85/min). The residual process used downstream is refitted on observations minus the filter's mean path (REBUILD step 3), not taken from this.

## REBUILD step 2 — framework merge and Phase 4 batch fit

Framework (`framework.tgz`) merged per INTERFACES.md: package files copied over `kmia2h/`, tests into `tests/`, `config.framework.yaml` appended to `config.yaml`. All 174 framework tests pass on the merged repo. Additions: `forcing.py` (live-only window and continuous forcing), `backtest/windows.py` (per-origin forcing + truth cache), `physics/batch_fit.py`, optional `c_scale` forcing (DECISIONS.md).

Batch fit, 2022-2023 hourly origins (15,261 windows), METAR sky-cover cloud fallback, KMIA METAR wind standing in for HRRR wind. All groups converge and sit inside bounds (`params/ode_prior.json`):

| Group | a (F/min per W/m2) | b (1/min) | c | kappa | lam | RMSE (F) | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 4.37e-5 | 5.6e-4 | 6.8e-5 | 0.91 | 0.00 | 1.73 | 15,261 |
| day / onshore | 7.58e-5 | 5.7e-3 | 2.1e-4 | 0.61 | 0.00 | 1.71 | 4,479 |
| day / offshore | 5.27e-5 | 0 | 1.3e-4 | 0.59 | 0.00 | 2.29 | 1,311 |
| day / other | 5.29e-5 | 0 | 0 | 0.93 | 1.00 | 2.51 | 1,926 |
| night / onshore | 5.6e-4 | 8.7e-4 | 1.0e-4 | 0.60 | 0.32 | 0.83 | 3,404 |
| night / offshore | 6.6e-4 | 0 | 1.8e-4 | 0.64 | 0.24 | 0.96 | 1,280 |
| night / other | 7.5e-4 | 9.5e-4 | 7.2e-5 | 0.51 | 0.07 | 1.05 | 2,861 |

Groups are by the origin's day/night, so "night" a is the sunrise heating of windows that start before dawn. Only day/onshore has a material relaxation to the upstream air (b = 0.0057/min, ~3 h); offshore and "other" fit b = 0, which says the lagged neighbour/buoy temperature (with METAR wind) carries no signal there that the trend does not.

2024, deterministic mean path from the last live KMIA observation (no filter, no noise), extremes + regime median gap:

| 2024 (n = 14,649) | Path RMSE vs smooth | Path RMSE vs 1-min | RMSE H_r | RMSE L_r |
| --- | --- | --- | --- | --- |
| **ODE mean path** | **1.76** | **1.84** | **1.91** | **1.95** |
| Persistence + trend | 2.73 | 2.78 | 2.24 | 2.48 |

Ablation (2024 RMSE of H_r / L_r with one term removed, `out/phase4/ablation.csv`): longwave matters most for the night low (dry night L 1.50 -> 2.02); removing cloud *improves* the daytime high (dry day H 2.64 -> 2.34, wet day 2.15 -> 2.05), so the held METAR cloud cover is a poor forecast of the next two hours' cloud; rain has no effect (no MRMS yet).

## REBUILD step 3 — earlier fixes, Phases 5-7

Applied: T_adv uncertainty as the filter state `dT_adv` plus the `T_m4` lag (framework state vector); residual refit on observations minus the filter's mean path; jump-detector efficiency correction; step check = 99.9th percentile of fitted jumps (7.29 F, warning only); inflation 1.01 with extra random walk on a, b and the upstream offset; one residual table shared by filter and simulator; spread calibration per season x day/night on 2022-2023 (Phase 8 driver).

**Residual refit** (`params/residual.json`, 78 regimes, X = 1-minute obs - filter mean T, 2016-2023, 3.78 M minutes). X has sd 1.27 F, with slow components (theta 0.04-0.10 /min, i.e. 10-25 min) that the Phase 3 decomposition had removed. Daytime regimes have a negative mean (-0.17 to -0.49 F): the filter's smooth T runs warm by day. 22 regimes have jumps (day, SCT/BKN). The detector efficiency on synthetic series was 1.5-5 (false detections from minute noise and whole-F steps), so the corrected rates are 0.05-0.4 per hour. Phase 6 KS against the Phase 3 floor fails in all regimes (`out/phase6/validation_vs_floor.csv`). This is expected: the two e_H measure different things (DECISIONS.md).

**Filter tuning on 2024** (`scripts/tune_filter.py`, 4 months, 15 settings): T noise 0.10 F/min, residual noise x 0.5.

**Phase 5 checks, 2024** (backtest `v3`, live streams only, 300 members):

| Check | Result | Target |
| --- | --- | --- |
| HFMETAR normalised innovation mean / variance / lag-1 | -0.005 / 0.887 / 0.184 | +-0.1 / 0.8-1.2 / < 0.2 — **pass** |
| METAR (tenths) innovation mean / variance / lag-1 | -0.042 / 1.57 / 0.214 | (not a done-when item; METAR sigma 0.3 F is too small) |
| Analysis RMSE of T at origin vs 1-minute truth | **0.885 F** vs last HFMETAR 1.020 F (vs last report of any kind 0.883 F) | lower than last HFMETAR — **pass** |
| Rank histogram chi2 p, by season x day/night | dry night 0.09; dry day, wet day, wet night < 0.001 | p > 0.01 or logged: by day mildly U-shaped (too narrow, ~13-15% in each end decile), wet night slightly biased |
| Filter run time | ~35 s per month filter-only (~7 min per year) | < 2 h per year — **pass** |

Before tuning (v2, Phase 3-derived noise): HFMETAR variance 0.64, ensemble-mean RMSE 0.95 F, hump-shaped rank histogram. Pass 1 with the untouched Phase 3 single-OU table: variance 2.0, lag-1 0.60.

**Phase 7**: one 4,000-path run takes 0.18 s; the same origin and seed give identical output; every forecast passes `simulate.validate_forecast` (tables sum to 1); `settle()` of the paths equals the rounded unrounded extreme exactly.

## REBUILD step 4 — Phase 8 without satellite data, scored on 2024

Backtest `v3`: continuous EnKF on live streams (KMIA HFMETAR/METAR, neighbours, VAKF1/FWYF1), METAR cloud fallback, no GOES / MRMS / HRRR; 2,000 paths per 30-minute origin, 2022-2024. Spread calibration and MOS are fitted on 2022-2023. Conformal and blend weights are fitted on 2024, as SPEC Phase 8 prescribes, so the 2024 rows for `mos` and `blend` are in-sample for those two steps. Floor and baseline are the Phase 3 oracle and persistence+trend on the same origins. Scores are mode hit rate / mean bin log-loss (probability floor 1e-4).

| 2024 | High hit | High log-loss | Low hit | Low log-loss | n |
| --- | --- | --- | --- | --- | --- |
| Floor (oracle) | 0.820 | 0.452 | 0.838 | 0.428 | 14,366 |
| Baseline | 0.294 | 2.043 | 0.268 | 2.132 | 14,221 |
| Raw ensemble | 0.352 | 1.925 | 0.294 | 2.144 | 14,535 |
| Spread-calibrated physics | 0.351 | 1.790 | 0.310 | 1.962 | 14,535 |
| MOS (no conformal) | 0.411 | 1.531 | 0.366 | 1.683 | 14,535 |
| MOS + conformal | 0.417 | 1.512 | 0.371 | 1.658 | 14,535 |
| **Blended (final)** | **0.419** | **1.496** | **0.375** | **1.650** | 14,535 |

**Against the first build** (floor 0.821/0.451, 0.836/0.425; baseline 0.293/2.054, 0.271/2.127; blended 0.404/1.520, 0.379/1.627):

- Floor and baseline reproduce within 0.003 in hit rate and 0.011 in log-loss. The small differences come from the origin set: this build scores the 30-minute backtest origins, and the first build's exact set is unknown.
- The blended high is a little better (hit +0.015, log-loss -0.024).
- The blended low is a little worse (hit -0.004, log-loss +0.023).

**Likely sources of the difference:**

- The filter noise was re-tuned on 2024 with a different grid.
- The batch-fit groups are by origin.
- The QC exclusion set differs (spikes are kept).

The model closes about 35% (high) and 28% (low) of the log-loss gap between baseline and floor.

Checks:

- Blended log-loss is below the raw ensemble's for both high and low. **Pass.**
- No crossed quantiles; all probability tables sum to 1.
- Conformal central-interval coverage on 2024: 90.0 / 79.9 / 60.0% for the high and 90.0 / 80.0 / 60.1% for the low, at nominal 90 / 80 / 60. Before conformal it was 90.1 / 78.4 / 58.7 and 88.7 / 76.8 / 58.0.
- Blend weights on MOS are 0.65-0.95 (all 0.90). No group hit 0 or 1.
- Reliability (2024, blended): |observed - predicted| < 0.05 in every decile with n > 100. Logged exceptions are sparse top deciles: high 0.6-0.7 (n = 801, +0.048), high 0.7-0.8 (n = 50, +0.068), low 0.6-0.7 (n = 42, -0.103), low 0.7-0.8 (n = 4).
- Raw-ensemble rank histogram of the realised H_r among path maxima (2024) is U-shaped: 20% in each outer decile. The raw ensemble is too narrow, and the calibration widens it by 1.3x at night and 1.6-1.8x by day. Day also gets a +0.2 F (high) and +0.6 F (low) shift.

## REBUILD step 5 — gridded downloads (lean plan), 2022-01-01 to 2024-12-31

| Source | What was read | Result | Wall time (8 vCPU, shared) |
| --- | --- | --- | --- |
| GOES-16 ABI L2 CONUS | ACMC + ACHAC at every 30-min origin plus one scan ~10 min earlier; CODC by day; HDF5 range reads of a +-100 km Miami box | 1,096 / 1,096 days, 52,455 origin rows (47.9 / day), station cloud fraction present 99.9%, cloud-top height 66.8%, optical depth 40.4% (day only) -> `store/goes_feat.parquet` (13 MB) | ~2.3 h (threads did not scale under h5py; switched to 10 processes) |
| HRRR | own `.idx` byte-range reader: TMP 2 m, UGRD/VGRD 10 m at leads 90-270 min (30-min steps) from wrfsubhf; TCDC entire atmosphere from hourly wrfsfc f01-f05, stored as 0-1 fraction | 26,304 / 26,304 cycles, 236,736 rows -> `store/nwp_point.parquet` (9 MB) | ~3.3 h (6 then 14 processes; network-latency bound) |
| MRMS MergedReflectivityQCComposite | file <= origin - 10 min and one ~10 min earlier, decoded in memory, sector/ring max / mean / frac >= 35 dBZ and motion | see step 6 | decode-bound (~0.85 s / file) |

No raw satellite, radar or NWP file is kept (0 `.nc` / `.grib2` files under `data/`). The disk in use is 3.4 GB of the 100 GB budget. No day was missing, so no `avail=False` day markers were needed. A source that is stale at an origin (GOES > 40 min, MRMS > 30 min, HRRR cycle > 6 h) gives `avail_<source> = 0` in the Phase 8 frame. Every wait loop waited for the downloader process to exit (`kill -0 <pid>`), not for a file count.

**GOES cloud-motion threshold, tuned on 2022-2023** (`params/cloud.json`). Station cloud fraction 30-120 min ahead, mean absolute error:

| Motion source | MAE |
| --- | --- |
| GOES motion whenever valid (q > 0, 96% of origins) — **chosen** | 0.175 |
| 1.5 x 10 m wind only | 0.203 |
| Persistence of the current station cloud fraction | 0.176 |

GOES advection barely beats persistence, so the advected cloud field carries little 2-hour skill at KMIA.
