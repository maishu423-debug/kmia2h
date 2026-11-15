"""Phase 10 regression check: the refactored backtest loop and frozen MOS reproduce the stored Phase 9
results. Runs 2025-02 through run_month into a scratch tag and re-scores 2025-01..02 through
phase9.table; both must equal the stored files exactly."""
import json
import numpy as np
import pandas as pd
from kmia2h.backtest import backtest
from kmia2h.config import CFG, out, store
from kmia2h.phases import phase9

FZ = CFG.frozen
USE = tuple(FZ.sources)
res = {}
backtest.run_month(("p10_check", 2025, 2, {"sources": USE, "overwrite": True, "save_paths": True}))
a = np.load(store("backtest/p10_check/202502.npz"))
b = np.load(store(f"backtest/p9_{FZ.backtest_tag}/202502.npz"))
for k in ("origin", "max_u", "min_u", "high", "low", "edge_high", "edge_low", "ensemble" if "ensemble" in a else "coef",
          "sample_paths", "rec_T", "inn_val"):
    res[f"backtest {k}"] = bool(np.array_equal(a[k], b[k], equal_nan=True))
t, _, _ = phase9.table(f"p9_{FZ.backtest_tag}", "2025-01", "2025-02", FZ.mos_name, USE)
s = pd.read_parquet(out(9, "per_origin_A.parquet"))
s = s[s.origin < pd.Timestamp("2025-03-01", tz="UTC")]
cols = [c for c in t.columns if c.endswith(("_logloss", "_hit", "_crps"))]
m = t.merge(s, on=["origin", "side"], suffixes=("", "_old"))
res["phase9 rows"] = [len(t), len(s)]
res["phase9 scores identical"] = bool(all(np.array_equal(m[c].to_numpy(), m[c + "_old"].to_numpy(), equal_nan=True) for c in cols))
print(json.dumps(res, indent=1))
out(10, "refactor_check.json").write_text(json.dumps(res, indent=1))
