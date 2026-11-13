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
