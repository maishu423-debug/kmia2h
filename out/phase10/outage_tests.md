# Phase 10 — outage tests

Live (spin-up) path at 60 random 2024 origins (validation year), each source removed for the 60 min before the origin. Every run must end in a forecast or ABSTAIN. Log-loss is scored against the 1-minute truth; the paired column is the mean change against the same origins with no outage. Nothing was tuned on these numbers.

| outage              |   runs |   ok |   abstain |   crash |   sec_median |   high_logloss |   low_logloss |   high_hit |   low_hit |   high_dlogloss_paired |   low_dlogloss_paired |
|:--------------------|-------:|-----:|----------:|--------:|-------------:|---------------:|--------------:|-----------:|----------:|-----------------------:|----------------------:|
| none                |     60 |   59 |         1 |       0 |        5.350 |          1.328 |         1.598 |      0.525 |     0.390 |                  0.000 |                 0.000 |
| goes                |     60 |   59 |         1 |       0 |        4.950 |          1.359 |         1.647 |      0.458 |     0.356 |                  0.031 |                 0.049 |
| mrms                |     60 |   59 |         1 |       0 |        5.600 |          1.440 |         1.601 |      0.441 |     0.542 |                  0.112 |                 0.004 |
| hrrr                |     60 |   59 |         1 |       0 |        5.300 |          1.328 |         1.609 |      0.508 |     0.339 |                  0.000 |                 0.011 |
| kmia                |     60 |    0 |        60 |       0 |        2.500 |        nan     |       nan     |    nan     |   nan     |                nan     |               nan     |
| goes+mrms+hrrr      |     60 |   59 |         1 |       0 |        5.200 |          1.539 |         1.637 |      0.305 |     0.525 |                  0.211 |                 0.040 |
| goes+mrms+hrrr+kmia |     60 |    0 |        60 |       0 |        2.700 |        nan     |       nan     |    nan     |   nan     |                nan     |               nan     |

Typical degradation notes:

- **goes**: goes: last scan 74 min old, used; cloud perturbation x2
- **goes+mrms+hrrr**: goes: last scan 74 min old, used; cloud perturbation x2 / mrms: missing; R = 0 and jump rate at the regime 90th percentile / hrrr: cycle 08Z used (newest expected 09Z), lead +1 h
- **goes+mrms+hrrr+kmia**: goes: last scan 74 min old, used; cloud perturbation x2 / mrms: missing; R = 0 and jump rate at the regime 90th percentile / hrrr: cycle 08Z used (newest expected 09Z), lead +1 h / kmia: newest report 60 min old -> ABSTAIN
- **hrrr**: hrrr: cycle 08Z used (newest expected 09Z), lead +1 h
- **kmia**: kmia: newest report 60 min old -> ABSTAIN
- **mrms**: mrms: missing; R = 0 and jump rate at the regime 90th percentile

Crashes: 0
