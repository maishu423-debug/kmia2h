# BLOCKED

| Date | Phase | Check | Result | Action |
| --- | --- | --- | --- | --- |
| 2026-11-13 | 2 | KMIA 1-minute coverage >= 90% in >= 90% of months since 2016 | 69.4% of months (75 of 108, 2016-2024) reach 90% after QC exclusion. These are real NCEI archive outages, not download failures (every monthly request returned 200; first build found 63% over 2016-2025). | Logged and continued (REBUILD.md). Windows with > 5 missing minutes are excluded from training and scoring. |
| 2026-11-13 | 5 | Rank histogram chi2 p > 0.01 by season x day/night | Dry night passes (p = 0.09); dry day, wet day and wet night fail (p < 0.001): mildly U-shaped by day, slightly biased at wet night. Analysis RMSE and innovation checks pass. | Logged (spec allows logging); the Phase 8 spread calibration and MOS correct the downstream spread. |
| 2026-11-13 | 6 | Simulated e_H / e_L match the Phase 3 quantiles per regime (KS < 0.05) | Fails in all four season x day/night groups (KS 0.49-0.60). The refit residual (REBUILD step 3) is relative to the filter path, not the centred 21-minute mean, so the two e_H differ by definition. | Logged; three distinct residual definitions were tried (Phase 3 acov fit, positive-lag fit, filter-path refit). The filter-path version was kept because it gives white innovations. |
| 2026-11-13 | 8 | Reliability |obs - pred| < 0.05 in every decile | Holds in every decile with n > 100; exceptions in sparse top deciles (n = 3-71). | Logged. |

## Phase 9 (2026-11-14)

- Abstain rule: 85% hit among acted-on origins is not achievable on 2024 at >= 1% acted on. The best achievable is high 79% and low 66%. The rule uses that, and it is evaluated unchanged on Test A and B (REPORT.md).
- The Test B vs Test A stability check flags the global final-model log-loss (+0.07 high, +0.08 low) and 154 of 648 stratified cells. The floor is stable, so the input streams are the suspect (2026 KMIA outages, no VAKF1 September 2026). Logged, not tuned.
- NDBC September 2026 is not yet archived (monthly file 404, before the realtime window), so avail_ndbc is False for 2026-09-01..29.

## Phase 10 (2026-11-15)

- The live KMIA HFMETAR latency through IEM is 17–29 min, against the 8 min assumed in training. Live runs will ABSTAIN more often than the backtest did under the 20-minute rule, and they see the station later. Logged. The aviationweather.gov METAR feed narrows the gap only around :53.
