"""Pre-Phase-9 changes, one experiment per call (fits on 2022-2023, scores 2024 only):
  python scripts/pre9.py metar   -> backtest s9_metar: s6_mrms + METAR sigma from 2022-2023 innovations
  python scripts/pre9.py sbday   -> + sea-breeze switch in daylight minutes only (window cache, batch fit, backtest)
Overrides go through KMIA2H_OVERRIDE so nothing else on disk changes; the ODE prior of each stage is
saved as params/ode_prior_<tag>.json."""
import json, os, shutil, sys, time
import pandas as pd

METAR_SIGMA = float(os.environ.get("METAR_SIGMA", "0.60"))
SRC = ("goes", "hrrr", "sea_breeze", "mrms")
stage = sys.argv[1]
ov = {"filter.obs_sigma.metar": METAR_SIGMA, "physics.upstream.sea_breeze_day_only": stage == "sbday"}
os.environ["KMIA2H_OVERRIDE"] = json.dumps(ov)
from kmia2h.backtest import backtest, windows  # noqa: E402
from kmia2h.config import out, path  # noqa: E402
from kmia2h.phases import phase4, quick_score  # noqa: E402

tag = f"s9_{stage}"
workers = int(os.environ.get("WORKERS", "8"))
t = time.time()
print(tag, ov, flush=True)
if stage == "sbday":
    if "--skip-windows" not in sys.argv:
        windows.build(tag, "2022-01", "2024-12", use=SRC, workers=workers)
        print("windows", round(time.time() - t), flush=True)
    if "--windows-only" in sys.argv:
        sys.exit(0)
    shutil.copy(path("params", "ode_prior_s6_mrms.json"), path("params", "ode_prior.json"))
    phase4.main(tag)
    shutil.copy(path("params", "ode_prior.json"), path("params", f"ode_prior_{tag}.json"))
    print("batch fit + phase4", round(time.time() - t), flush=True)
else:
    shutil.copy(path("params", "ode_prior_s6_mrms.json"), path("params", "ode_prior.json"))
    shutil.copy(path("params", "ode_prior.json"), path("params", f"ode_prior_{tag}.json"))
backtest.run(tag, "2022-01", "2024-12", workers=workers, sources=SRC, overwrite=True)
print("backtest", round(time.time() - t), flush=True)
q = quick_score.main(tag)
(out(6, "x").parent.parent / "pre9").mkdir(exist_ok=True)
q.to_csv(out(6, "x").parent.parent / "pre9" / f"score_{tag}.csv", index=False)
pd.set_option("display.width", 200)
print(q.round(3).to_string(index=False))
print("took", round(time.time() - t))
