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
