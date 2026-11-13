The previous machine was lost. Rebuild from SPEC.md, reusing what we already learned (below). Do not ask me questions. After every commit, push: git push -u origin HEAD.

This machine: Azure D8s_v6, 8 vCPU, 32 GB RAM, 128 GB disk. Disk budget 100 GB.

Known results from the first build (reproduce or explain differences):
- Settlement rule: rolling 5-minute average of whole-F 1-minute values, rounded half_up. Matched CLI on 99.46% of highs, 98.07% of lows (2020-2025). Raw 1-minute max/min is wrong.
- 1-minute coverage has real NCEI outages (only 63% of months >=90% coverage). Log in BLOCKED.md and continue.
- Settled high sits ~0.29 F above the smooth path's max, low ~0.23 F below. Define edge after adding the regime's median gap.
- Exclude QC-flagged 1-minute readings everywhere (e.g. a 122 F stretch in July 2023).
- 2024 scores to reproduce (hit / log-loss): floor high 0.821/0.451, low 0.836/0.425; baseline high 0.293/2.054, low 0.271/2.127; blended model high 0.404/1.520, low 0.379/1.627.

Steps:
1. Build-order steps 1-3 (rounding rule, station data, noise floor).
2. Extract framework.tgz and merge it following framework/INTERFACES.md. Write physics/batch_fit.py. Use buoy VAKF1 (FWYF1 stopped in 2022). HRRR cloud as a 0-1 fraction.
3. Apply the earlier fixes: T_adv as a state plus the T_m4 lag; residual refit on observations minus the filter's mean path; jump-detector efficiency correction; step-check threshold = 99.9th percentile of fitted jumps (warning only); filter inflation ~1.01 with extra random-walk noise on a, b and the upstream term; filter and simulator share one set of noise settings; spread calibration per season x day/night on 2022-2023.
4. Run Phase 8 (MOS + conformal + blend) without satellite data and score 2024. Write the summary against the numbers above into REPORT.md.
5. Then start build-order step 4 in the background, lean plan: GOES at 30-minute origins plus one scan 10 min earlier, CODC daytime only, Miami-subset range reads (~0.36 MB/file); MRMS at the same times; HRRR leads 90-270 min via your own .idx byte-range reader, total cloud from hourly files. Tune the GOES cloud-motion threshold on 2022-2023. Wait loops must wait for the downloader process to exit, not count files; mark missing days avail=False.
6. When downloads finish, add GOES cloud advection, HRRR, sea-breeze switch and MRMS rain in that order, re-scoring 2024 after each; re-tune daytime noise; rerun Phase 8; report by day/night and wet/dry season.

Never use 2025-2026 for fitting or scoring. Commit and push after each step. Stop after step 6 and summarise.