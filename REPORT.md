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
