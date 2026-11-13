"""Phase 8 on a backtest tag: spread calibration, MOS, conformal, blend; writes out/phase8/."""
import json, pickle, sys, time
import numpy as np, pandas as pd
from kmia2h.config import out
from kmia2h.mos import pipeline

tag = sys.argv[1]
t = time.time()
use = next((a.split('=', 1)[1].split(',') for a in sys.argv[2:] if a.startswith('--sources=')), [])
sc, st, report, d, res = pipeline.main(tag, reuse='--reuse' in sys.argv, use=tuple(use))
sc.to_csv(out(8, f"scores_{tag}.csv"), index=False)
st.to_csv(out(8, f"scores_stratified_{tag}.csv"), index=False)
json.dump(report, open(out(8, f"report_{tag}.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
with open(out(8, f"probs_{tag}.pkl"), "wb") as f:
    pickle.dump({"origin": d.origin.to_numpy(), "season": d.season.to_numpy(), "is_day": d.is_day.to_numpy(),
                 "sector": d.sector.to_numpy(), "rain_recent": d.rain_recent.to_numpy(),
                 **{f"{s}_{k}": v for s in res for k, v in res[s].items()}}, f)
pd.set_option("display.width", 200)
print(sc.round(3).to_string(index=False))
for s in ("high", "low"):
    print(s, "spread", report[s]["spread_scale"], "blend", report[s]["blend_weights"])
    print("  coverage conformal", report[s]["coverage_conformal"], "raw MOS", report[s]["coverage_raw_mos"], "crossed", report[s]["crossed"])
    print("  rank hist 2024", report[s + "_rank_hist"]["2024"], "calibrated", report[s + "_rank_hist_calibrated"]["2024"])
print("took", round(time.time() - t))
