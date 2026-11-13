"""REBUILD step 6, one addition per call: python scripts/step6.py STAGE
STAGE in goes | hrrr | seabreeze | mrms. Builds the window cache with the cumulative sources,
refits the ODE prior (batch fit, kappa_thin, b_onshore), reruns the 2022-2024 backtest and
re-scores 2024 (physics, spread-calibrated on 2022-2023)."""
import json, shutil, sys, time
import pandas as pd
from kmia2h.backtest import backtest, windows
from kmia2h.config import out, path
from kmia2h.phases import phase4, quick_score

STAGES = {"goes": ("goes",), "hrrr": ("goes", "hrrr"), "seabreeze": ("goes", "hrrr", "sea_breeze"),
          "mrms": ("goes", "hrrr", "sea_breeze", "mrms")}
stage = sys.argv[1]
use = STAGES[stage]
tag = f"s6_{stage}"
t = time.time()
if stage == "goes":
    import subprocess
    subprocess.run(["uv", "run", "python", "scripts/tune_goes_motion.py"], check=True)
if "--skip-windows" not in sys.argv:
    windows.build(tag, "2022-01", "2024-12", use=use)
print("windows", round(time.time() - t), flush=True)
r, a = phase4.main(tag)
shutil.copy(path("params", "ode_prior.json"), path("params", f"ode_prior_{tag}.json"))
print("batch fit + phase4", round(time.time() - t), flush=True)
if stage == "mrms":
    from kmia2h.physics import rain_fit
    res = rain_fit.main()
    print("rain fit", res[0] if res else None, flush=True)
backtest.run(tag, "2022-01", "2024-12", sources=use, overwrite=True)
print("backtest", round(time.time() - t), flush=True)
q = quick_score.main(tag)
q.to_csv(out("step6", f"score_{tag}.csv"), index=False)
pd.set_option("display.width", 200)
print(q.round(3).to_string(index=False))
print("took", round(time.time() - t))
