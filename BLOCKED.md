# BLOCKED

| Date | Phase | Check | Result | Action |
| --- | --- | --- | --- | --- |
| 2026-11-13 | 2 | KMIA 1-minute coverage >= 90% in >= 90% of months since 2016 | 69.4% of months (75 of 108, 2016-2024) reach 90% after QC exclusion. These are real NCEI archive outages, not download failures (every monthly request returned 200; first build found 63% over 2016-2025). | Logged and continued (REBUILD.md). Windows with > 5 missing minutes are excluded from training and scoring. |
