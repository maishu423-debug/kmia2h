"""Fourth pre-Phase-9 combination: daytime-only sea breeze with the old METAR sigma (0.3). Reuses the
s9_sbday batch fit (window forcing does not depend on the filter's METAR sigma)."""
import json, os, shutil, time
os.environ["KMIA2H_OVERRIDE"] = json.dumps({"filter.obs_sigma.metar": 0.3, "physics.upstream.sea_breeze_day_only": True})
import pandas as pd
from kmia2h.backtest import backtest
from kmia2h.config import out, path
from kmia2h.phases import quick_score
tag = "s9_sb03"
shutil.copy(path("params", "ode_prior_s9_sbday.json"), path("params", "ode_prior.json"))
shutil.copy(path("params", "ode_prior.json"), path("params", f"ode_prior_{tag}.json"))
t = time.time()
backtest.run(tag, "2022-01", "2024-12", workers=int(os.environ.get("WORKERS", "6")),
             sources=("goes", "hrrr", "sea_breeze", "mrms"), overwrite=True)
q = quick_score.main(tag)
q.to_csv(out(6, "x").parent.parent / "pre9" / f"score_{tag}.csv", index=False)
print(q.round(3).to_string(index=False)); print("took", round(time.time() - t))
