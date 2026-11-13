# Decisions

| Date | Phase | Decision | Reason |
| --- | --- | --- | --- |
| 2026-11-13 | Setup | Rebuild on a fresh machine from SPEC.md + REBUILD.md. Disk budget 100 GB (REBUILD.md overrides SPEC's 150 GB). | REBUILD.md. |
| 2026-11-13 | Setup | 2025-2026 are never used for fitting or scoring. 1-minute data is downloaded through 2025 only for the Phase 1 rule check (as in the first build); METAR / neighbour downloads stop at 2024-12-31. | REBUILD.md: "Never use 2025-2026 for fitting or scoring." |
| 2026-11-13 | 2 | IEM station ids are the 3-letter form (`MIA`); `?help` showed `KMIA` is rejected. All IEM requests go through one sequential client, >= 2 s apart. The 11 one-off CLI requests ran beside the long download job. | SPEC Phase 2 table asks to confirm the id format. |
| 2026-11-13 | 2 | IEM's CSV has no report-type column when types are combined; an HFMETAR row is identified by the `MADISHF` tag in its raw text. HFMETAR temperature is the whole-degC main group (IEM leaves `tmpf` empty for these rows). | Observed in the raw output. |
| 2026-11-13 | 1 | Rounding rule `half_up`. half_up, half_even and via_c tie exactly (no four-valid .5 tie decided a daily extreme); ties go to the first rule listed, half_up. | SPEC Phase 1 step 3. |
| 2026-11-13 | 1 | Rolling 5-minute averages, not raw 1-minute extremes: raw max/min match only 31.7% / 68.2% of days against 99.5% / 98.1%. | SPEC Phase 1 step 4. |
| 2026-11-13 | 2 | QC flags were tightened, and only RANGE and FLAT are excluded (`qc.exclude_mask = 6`). FLAT needs temperature and dewpoint both unchanged > 45 min in daylight; the STEP bit marks an isolated one-minute spike and is stored but not excluded. | The literal step rule flagged real rain-cooled drops; single-minute downward spikes are demonstrably counted by the official CLI (e.g. 2020-04-10: 76,76,76,76,69 -> 74.6 -> CLI low 75). Excluding them lowered the CLI low match from 98.1% to 97.2%. The 122 F stretch of 2023-07-16/17 is removed by RANGE. |
