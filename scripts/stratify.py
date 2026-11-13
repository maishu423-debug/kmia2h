"""2024 scores by day/night x wet/dry season for Phase 8 runs, with floor and baseline on the same
origins. Usage: stratify.py TAG [TAG ...] -> out/step6/stratified.csv"""
import pickle, sys
import numpy as np, pandas as pd
from kmia2h.config import out
from kmia2h.mos import pipeline

rows = []
fb = None
for tag in sys.argv[1:]:
    P = pickle.load(open(out(8, f"probs_{tag}.pkl"), "rb"))
    origin = pd.DatetimeIndex(P["origin"])
    va = origin >= pd.Timestamp("2024-01-01", tz="UTC")
    grp = np.where(P["season"] == "wet", "wet", "dry").astype(object) + "-" + np.where(P["is_day"].astype(bool), "day", "night")
    if fb is None:
        d = pd.DataFrame({"origin": origin[va]})
        fb = pipeline.floor_baseline(d)
    for side in ("high", "low"):
        real = P[f"{side}_realised"]
        for model in ("raw", "physics", "blend"):
            p = P[f"{side}_{model}"]
            for g in ["all", *sorted(set(grp))]:
                m = va & ((grp == g) if g != "all" else True)
                rows.append({"tag": tag, "side": side, "model": model, "group": g, **pipeline.score(p[m], real[m])})
        if tag == sys.argv[1]:
            gi = pd.Series(grp[va], index=origin[va])
            for name in ("floor", "baseline"):
                s = fb[f"{name}_{side}"]
                gg = gi.reindex(s.index)
                for g in ["all", *sorted(set(grp))]:
                    ss = s if g == "all" else s[gg.to_numpy() == g]
                    rows.append({"tag": "-", "side": side, "model": name, "group": g, "hit": float(ss.mode_hit.mean()),
                                 "logloss": float(ss.logloss.mean()), "n": int(len(ss))})
r = pd.DataFrame(rows)
r.to_csv(out(6, "x").parent.parent / "step6" / "stratified.csv", index=False)
pd.set_option("display.width", 220)
piv = r.pivot_table(index=["side", "group"], columns=["model", "tag"], values="logloss")
print(piv.round(3).to_string())
piv = r.pivot_table(index=["side", "group"], columns=["model", "tag"], values="hit")
print(piv.round(3).to_string())
