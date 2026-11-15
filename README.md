# kmia2h

Probabilistic 2-hour high / low forecasts for KMIA (Miami International) on the Kalshi settlement rule. Design: `SPEC.md`; results: `REPORT.md`.

```
uv sync
uv run python run.py --origin now --window-start day
```

## Live bundle

A fresh clone can run live forecasts (`run.py --origin now`) without any training data. Git holds only what the frozen pipeline reads at run time:

| File | Content |
| --- | --- |
| `config.yaml` | All constants; `frozen:` names the configuration (`s9_sb03`) |
| `params/*.json` | Fitted parameters: ODE prior, filter and residual settings, cloud and rain, spread calibration, conformal offsets and blend weights (`conformal_s9_sb03.json`), abstain rule (`abstain.json`), elapsed-day pmf (`elapsed_obs.json`), station metadata |
| `models/mos_high_s9_sb03.joblib` | Frozen MOS boosters, high (18.3 MB; joblib, zlib level 3) |
| `models/mos_low_s9_sb03.joblib` | Frozen MOS boosters, low (11.4 MB) |

The models are the same pickle stream as the uncompressed `.pkl` training output, compressed losslessly. `pipeline.load_model` reads the `.joblib` when it is present and falls back to the `.pkl`, and the forecasts are byte-identical either way (checked on all of 2024 and on `run.py` runs). Every bundle file is under GitHub's 50 MB limit, so plain git is enough and no Git LFS is needed.

Git does not hold the training data, feature stores (`data/`), backtests, other model versions, the uncompressed `.pkl` files, the per-quantile `.lgb` copies, logs, or live output (`out/live/`). `.gitignore` excludes `models/*` and lists the bundle files as explicit exceptions. No credentials are needed: every live source is public. A clean clone has no persisted filter state, so its first run spins up from the last 6 h.

`--mode replay` and `--mode spinup` (historical origins) need the local stores and are not part of the bundle.

**Updating after retraining**

1. Train and freeze as before: a new backtest tag and Phase 8 run (`scripts/run_phase8.py`) write `models/mos_{high,low}_<name>.pkl`, the compressed `models/mos_{high,low}_<name>.joblib` and `params/conformal_<name>.json`; the ODE prior goes to `params/ode_prior.json`.
2. Point `frozen:` in `config.yaml` at the new tag and name; refit `params/abstain.json` on the validation year if the model changed.
3. In `.gitignore`, replace the two `!/models/mos_*_<old>.joblib` exceptions with the new names. Check sizes with `find config.yaml params models/*.joblib -size +50M`: anything over 50 MB will not push to GitHub as a plain file.
4. Scan before committing: `strings -n 8 <bundle files> | grep -Ei 'token|secret|password|api.?key|BEGIN .*PRIVATE KEY'` should print nothing.
5. Commit and push, then check from a fresh clone: `git clone … && uv sync && uv run python run.py --origin now`.
