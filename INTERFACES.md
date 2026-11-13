# Framework inputs

Everything the framework (SPEC.md Phases 4-8) expects from the data side. All times are UTC and timezone-aware. Temperatures are °F, cloud is a fraction 0-1, winds and motion vectors are m/s, and minute indices are whole minutes.

**Rules for every input**

- A row may be used only if `available_at <= origin`. `upstream.build` applies this itself. For all other inputs the caller must have applied it.
- NaN means missing. Missing optional inputs follow the fallbacks given below.
- Forecast-minute series have one value per minute, and row `i` is valid at minute `origin + i + 1`. HRRR at 15-minute steps must be interpolated to one minute first.

## Merging into the main repo

1. Copy `framework/kmia2h/` over `kmia2h/kmia2h/`. This adds new files and replaces `physics/solar.py`, which keeps `location()`, `compute()` and `cached()` for `features.py` and `data/qc.py`.
2. Copy `framework/tests/*` into `kmia2h/tests/`, and copy `framework/config.framework.yaml` beside `kmia2h/config.yaml`.
3. Append `config.framework.yaml` to `config.yaml`. Every top-level key in it is new.
4. Pass `ops=kmia2h.target` to `simulate.run`, and `cfg=CFG` everywhere else. `CFG` is a dict subclass, so it works as it stands.

Replace every value marked PLACEHOLDER in the config once the batch fit and 2024 tuning exist.

## Interfaces in `kmia2h/interfaces.py`

| Protocol | Provided by | Members |
| --- | --- | --- |
| `Target` | `kmia2h.target` module | `settle(t1, rule)`, `round_rule(x, rule)`, `edge(x)` |
| `Window` | `window.py`, when it exists | `elapsed_min(origin)`, `remaining_min(origin)`, as integers |
| `FeatureProvider` | `features.py` | `features_at(origins)` |

`settle` must accept any length `4 + w`, not only 124, so that elapsed windows work. The main repo's `target.settle` does.

## Config

Main-repo keys read: `target.rounding`, `target.roll_window`, `sim.horizon_min`, `sim.lead_min`, `sim.paths_live`, `sim.seed`, `features.sector_onshore`, `features.sector_offshore`, `features.wet_months`, `neighbours`, `tz_lst_offset_hours`. Everything else is in `config.framework.yaml`.

## Phase 4: forcing

### Site metadata

`site` and `kmia`: mapping with `lat`, `lon` (degrees) and `elevation_m`. `meta`: mapping of station id to `{lat, lon}`, covering neighbours and buoys. The shape of `params/stations.json` already fits. **Buoys FWYF1 and VAKF1 must be added**, as that file holds ASOS stations.

### `solar.clear_sky(index, site)`

| Input | Type | Resolution |
| --- | --- | --- |
| `index` | `DatetimeIndex` | 1 minute |

Returns `ghi` (W/m², becomes `s_clr`), `zenith`, `elevation`, `is_day`.

### `upstream.build(origin, n, obs, kmia_obs, meta, kmia, u10, v10, cfg, t_last, hrrr_t2m, hrrr_dir)`

| Input | Columns or type | Units and notes | Resolution |
| --- | --- | --- | --- |
| `obs` | `station`, `ts_utc`, `tmpf`, `available_at` | °F. Neighbours from `obs.parquet`. Buoys from `ndbc.parquet` with `atmp_f` renamed `tmpf`. | as reported |
| `kmia_obs` | `ts_utc`, `drct`, `dwpf`, `available_at` | degrees, °F. Used for the sea-breeze test. At least the last 30 minutes. | as reported |
| `u10`, `v10` | float | HRRR 10 m wind at origin, m/s | one value |
| `t_last` | float | Last KMIA temperature, the last fallback | one value |
| `hrrr_t2m` | array (n) or None | HRRR 2 m T at the forecast minutes, °F | 1 minute |
| `hrrr_dir` | array (n) or None | HRRR 10 m direction, degrees from which wind blows | 1 minute |

Fallback at each minute: upwind stations, then HRRR 2 m T, then `t_last`. Returns `t_adv`, `b_scale`, `onset`, `wind_class`.

### `cloud_advect.build(valid, goes_row, u10, v10, is_day, sky_code, cfg, tcdc, tcdc_origin)`

`goes_row` is one mapping, or None to use the METAR fallback.

| Key | Units and notes |
| --- | --- |
| `cf_disc` | Cloud fraction over the 5 km station disc, 0-1 |
| `cf_s{SS}_r{R}` | Cloud fraction, sector `SS` = 00-11, ring `R` = 0-3 |
| `cod_disc` | Cloud optical depth, NaN at night |
| `cth_disc` | Cloud-top height, **metres** |
| `motion_u`, `motion_v`, `motion_q` | Cloud motion (m/s, direction of travel) and quality score |

Sector `k` covers bearings `[30k, 30(k+1))` degrees clockwise from north, looking outward from KMIA. Rings are 0-10, 10-20, 20-40 and 40-80 km.

| Other input | Units and notes |
| --- | --- |
| `valid` | `DatetimeIndex` of the n forecast minutes |
| `sky_code` | Dominant METAR cover: CLR, SKC, NCD, FEW, SCT, BKN, OVC or VV |
| `tcdc`, `tcdc_origin` | HRRR total cloud as a **fraction**. HRRR delivers percent: divide by 100. Percent raises an error. |

Returns `c`, `kappa_scale`, `source`, `u`, `v`.

**Naming note.** `features.py` prefixes these columns with `goes_` (for example `goes_cf_disc`). Strip the prefix when building `goes_row`.

### `rain.find_cells(row, n, cfg)`

`row` is one mapping from MRMS, or None.

| Key | Units and notes |
| --- | --- |
| `maxdbz_s{SS}_r{R}` | Max reflectivity in the sector and ring, dBZ |
| `frac35_s{SS}_r{R}` | Fraction of pixels at or above 35 dBZ, 0-1 |
| `motion_u`, `motion_v` | Cell motion, m/s, direction of travel |

Returns a frame of `arrival_min`, `drop_f`, `sector`, `ring`, which is what `rain.rate` and `simulate.run(rain_cells=...)` take.

### `ode.integrate(t0, coef, forcing, cfg)` and the forcing table

`forcing` is a mapping or DataFrame of 1-D columns of equal length with no NaN:

| Column | Units | Comes from |
| --- | --- | --- |
| `s_clr` | W/m² | `solar.clear_sky(...)["ghi"]` |
| `c` | 0-1 | `cloud_advect.build(...)["c"]` |
| `t_adv` | °F | `upstream.build(...)["t_adv"]` |
| `t_sky` | °F | `ode.sky_temperature_f(t, dewpoint, cfg)` |
| `r` | °F/min | `rain.rate(...)` |
| `b_scale` | ratio, optional | `upstream.build(...)["b_scale"]` |
| `kappa_scale` | ratio, optional | `cloud_advect.build(...)["kappa_scale"]` |

`coef`: `a` (°F/min per W/m²), `b` (1/min), `c` (°F/min per W/m²), `kappa`, `lam`. These come from `params/ode_prior.json` once the batch fit exists.

## Phase 5: filter

| Input | Columns or type | Units and notes | Resolution |
| --- | --- | --- | --- |
| Filter forcing | `s_clr`, `c`, `t_adv`, `t_sky`, `r`, optional `b_scale`, `kappa_scale` | As above, indexed by filter minute. `c` may be NaN, and C_eff then persists. Built from live streams only. | 1 minute, continuous |
| Regime series | `theta_f`, `sigma_f`, `theta_s`, `sigma_s` | From `process_model.regime_series(table, season, is_day, sky_class, wind)` | 1 minute |
| Observations | `ts_utc`, `available_at`, `kind`, `value` | See below. Convert with `obs_operators.obs_from_frame(df, start)`. | as reported |

| `kind` | `value` | Source | Observation error (config) |
| --- | --- | --- | --- |
| `hfmetar` | °F converted from whole °C | `report_type = 1` | 0.52 °F |
| `metar` | °F from the tenths-°C T group | `report_type = 3, 4` | 0.3 °F |
| `goes` | cloud fraction over KMIA | GOES `cf_disc` | 0.15 |
| `hrrr` | HRRR 2 m T at the valid time | HRRR | 1.5 °F |

KMIA 1-minute data must never appear here. Upstream temperature enters as forcing, not as an observation (see `DECISIONS.md`).

Filter output, `EnKF.snapshot()`: `ensemble` (N × 14, in `process_model.STATE` order), `paths` (N × up to 120, the smoothed sensor temperature in °F, oldest first, ending at the snapshot minute), and `k`.

## Phase 6: residual fit

`residual_fit.fit_regime(x, acov, is_day, sky_class, hour, cfg)`, once per regime:

| Input | Type | Units and notes | Resolution |
| --- | --- | --- | --- |
| `x` | array | `t1 - m` from Phase 3, °F, NaN where missing | 1 minute |
| `acov` | mapping | `A_f`, `theta_f`, `A_s`, `theta_s`, `white`, a row of `params/residual_acov.json`. `white` includes the 1/12 °F² of quantisation. | one per regime |
| `is_day` | bool array | Same length as `x` | 1 minute |
| `sky_class` | str array | `clear`, `sct`, `bkn`, `ovc` or `unk` | 1 minute |
| `hour` | int array | Local standard hour | 1 minute |

Regime tables use the Phase 3 key format: `wet|True|sct|5to10`, then `wet|True|sct`, then `wet|True`.

Fitted fields: `theta_f`, `sigma_f`, `theta_s`, `sigma_s`, `white`, `lam_per_hour`, `lam_by_sky_hour`, `active_fraction`, `p_drop`, `mean_drop`, `mean_rise`, `jump_min`, `excess_kurtosis`, `nu`. The sigmas are SDE values, °F per root minute.

## Phase 7: simulation

`simulate.run(snapshot, forcing, rp, ops, cfg, origin, seed, n_paths, elapsed_min, regime, latest_obs, floor_range, cloud, rain_cells)`

| Input | Notes |
| --- | --- |
| `snapshot` | The filter's `snapshot()`. `paths` needs at least `elapsed_min + 4` minutes when `elapsed_min > 0`. |
| `forcing` | The forcing table for the remaining window. Its length sets the window. |
| `rp` | Fitted regime parameters, scalars or one value per minute. `lam_per_hour` must be 0 at night and outside SCT/BKN. |
| `ops` | `kmia2h.target` |
| `cloud` | Optional: `cf_disc`, `cf_field` (12 × 4 array from `cloud_advect.field_from_row`), `u`, `v`, optional `tcdc` |
| `rain_cells` | Optional output of `rain.find_cells` |
| `latest_obs`, `floor_range` | Optional inputs of the sanity checks |

Output: `SimResult`. `forecast` has SPEC.md's schema, and `high`, `low`, `max_unrounded`, `min_unrounded`, `paths`, `checks` and `forcing_used` hold the arrays and diagnostics.

## Phase 8: MOS and calibration

### Training frame for `train.train(df, target, cfg)`

One row per origin (30-minute cadence). `target` is `h_r` or `l_r`.

| Column | Units and source | Required |
| --- | --- | --- |
| `origin` | UTC time | Yes |
| `h_r`, `l_r` | Unrounded realised max and min of the rolling 5-minute mean over the window, °F, from 1-minute data | Yes, the target |
| `q_max_05, _25, _50, _75, _95` | `forecast["q_max"]` | Yes, for `h_r` |
| `q_min_05, _25, _50, _75, _95` | `forecast["q_min"]` | Yes, for `l_r` |
| `edge_high`, `edge_low` | `forecast` | Optional |
| `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` | `features.py` | Optional |
| `ens_spread` | `SimResult.checks["spread_of_path_max"]`, °F | Optional |
| `t_last`, `t_age_min`, `trend_30` | `features.py`, °F, minutes, °F | Optional |
| `t_adv_anom` | Mean `t_adv` over the window minus `t_last`, °F | Optional |
| `cf_disc` | GOES station cloud fraction, 0-1 | Optional |
| `max_dbz_up` | Max upstream reflectivity, dBZ | Optional |
| `hrrr_t2m_end` | HRRR 2 m T at window end, °F (`hrrr_t2m_p120`) | Optional |
| `regime_*` | One-hot regime columns, 0 or 1 | Optional |
| `avail_*` | Source flags, 0 or 1 | Optional |

Optional columns may be absent: they become NaN, which LightGBM handles. Prediction frames need the same columns, without the target.

### Calibration and blending

| Function | Inputs |
| --- | --- |
| `conformal.fit(q, y, taus, groups, cfg)` | `q` (rows × taus) from `predict.predict_quantiles` on 2024, `y` the realised `h_r` or `l_r`, `groups` from `conformal.group_key(season, is_day)` |
| `conformal.fit_blend(p_mos, p_physics, realised, groups, cfg)` | Two arrays (rows × bins) on the same integer bins (`predict.on_bins`), and `realised` the index of the settled bin in each row |
| `predict.joint_table(ks_high, p_high, ks_low, p_low, rho, cfg)` | Marginals, and `rho = predict.copula_rho(max_unrounded, min_unrounded, rho_max)` from `SimResult` |

Nothing in the framework reads or writes a file, except the main-repo `solar.cached()` and its config lookups. Models and parameters are returned to the caller to save.
