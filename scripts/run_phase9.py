"""Phase 9 driver. Reads the frozen configuration from config.yaml (`frozen:`) and never fits anything
except the abstain rule, on 2024.

  python scripts/run_phase9.py prep       rebuild the stores through periods.data_end (test periods added)
  python scripts/run_phase9.py backtest   window targets (v1) and the frozen backtest for Test A + Test B
  python scripts/run_phase9.py score      abstain rule on 2024, scores on Test A and B, stability, worst 50
"""
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

from kmia2h.config import CFG, out, path, store

FZ = CFG.frozen
TEST_FIRST, TEST_LAST = "2025-01", pd.Timestamp(CFG.periods.data_end).strftime("%Y-%m")
TEST_TAG = f"p9_{FZ.backtest_tag}"
USE = tuple(FZ.sources)


def digest(df: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def pre2025(name: str, col: str | None = None) -> str:
    d = pd.read_parquet(store(name))
    cut = pd.Timestamp("2025-01-01", tz="UTC")
    d = d[d[col] < cut] if col else d[d.index < cut]
    return digest(d.reset_index(drop=col is not None))


def prep() -> None:
    from kmia2h.data import fetch_goes, fetch_iem_1min, fetch_mrms, fetch_ndbc, fetch_nwp, parse_obs
    from kmia2h.features import build_grids
    from kmia2h.floor import noise_floor
    checks = {"kmia_1min.parquet": None, "obs.parquet": "ts_utc", "ndbc.parquet": "ts_utc",
              "goes_feat.parquet": "ts_utc", "mrms_feat.parquet": "ts_utc", "nwp_point.parquet": "valid_utc",
              "features_30m.parquet": None, "features_10m.parquet": None}
    before = {k: pre2025(k, c) for k, c in checks.items()}
    t = time.time()
    fetch_iem_1min.build(); print("1min", round(time.time() - t), flush=True)
    parse_obs.build(); print("obs", round(time.time() - t), flush=True)
    fetch_ndbc.main(); print("ndbc", round(time.time() - t), flush=True)
    fetch_goes.build(); fetch_mrms.build(); fetch_nwp.build(); print("gridded", round(time.time() - t), flush=True)
    build_grids(end=CFG.periods.data_end); print("features", round(time.time() - t), flush=True)
    ft = noise_floor.extend_table(); print("floor table", round(time.time() - t), flush=True)
    after = {k: pre2025(k, c) for k, c in checks.items()}
    same = {k: before[k] == after[k] for k in checks}
    old = pd.read_parquet(store("floor_table_L21.parquet"))
    new = ft.reindex(old.index)[old.columns]
    same["floor_table_L21 (<=2024)"] = bool(np.allclose(old.select_dtypes("number").to_numpy(float),
                                                         new.select_dtypes("number").to_numpy(float), equal_nan=True)
                                            and (old.regime == new.regime).all())
    print(json.dumps(same, indent=1))
    out(9, "prep_check.json").write_text(json.dumps(same, indent=1))


def run_backtest() -> None:
    from kmia2h.backtest import backtest, windows
    t = time.time()
    windows.build("v1", TEST_FIRST, TEST_LAST, use=())
    print("windows", round(time.time() - t), flush=True)
    backtest.run(TEST_TAG, TEST_FIRST, TEST_LAST, sources=USE, overwrite=True, save_paths=True)
    print("backtest", round(time.time() - t), flush=True)


def md(df: pd.DataFrame, cols) -> str:
    return df[cols].to_markdown(index=False, floatfmt=".3f")


def score() -> None:
    from kmia2h.phases import phase9
    t0 = time.time()
    name = FZ.mos_name
    t24, res24, _ = phase9.table(FZ.backtest_tag, "2024-01", "2024-12", name, USE)
    rule = phase9.fit_abstain(t24)
    path("params", "abstain.json").write_text(json.dumps(rule, indent=1))
    print("abstain", {s: {k: rule[s][k] for k in ("threshold", "hit_2024", "share_2024", "target_reached")} for s in ("high", "low")}, flush=True)
    periods = {"A": ("2025-01", "2025-12"), "B": ("2026-01", TEST_LAST)}
    tabs, res, summ, strat = {}, {}, {}, {}
    for k, (a, b) in periods.items():
        tabs[k], res[k], _ = phase9.table(TEST_TAG, a, b, name, USE)
        tabs[k].to_parquet(out(9, f"per_origin_{k}.parquet"))
        summ[k] = phase9.summarise(tabs[k])
        strat[k] = pd.concat([phase9.summarise(tabs[k], by) for by in ("hour_lst", "season", "sector", "sky_class",
                                                                       "rain_win", "edge_bin", "is_day")])
        summ[k].to_csv(out(9, f"scores_{k}.csv"), index=False)
        strat[k].to_csv(out(9, f"scores_stratified_{k}.csv"), index=False)
        print(k, "scored", round(time.time() - t0), flush=True)
    s24 = phase9.summarise(t24)
    s24.to_csv(out(9, "scores_2024_frozen.csv"), index=False)
    ab = {k: phase9.evaluate_abstain(rule, tabs[k]) for k in periods}
    ab24 = phase9.evaluate_abstain(rule, t24)
    stab = phase9.stability(pd.concat([summ["A"], strat["A"]]), pd.concat([summ["B"], strat["B"]]))
    stab.to_csv(out(9, "stability.csv"), index=False)
    rel = {k: phase9.reliability(tabs[k], res[k]) for k in periods}
    rh = {k: phase9.rank_hist(tabs[k]) for k in periods}
    worst = phase9.worst_windows(tabs["A"], TEST_TAG, 50, USE)
    print("worst", round(time.time() - t0), flush=True)

    cols = ["side", "model", "n", "logloss", "logloss_se", "hit", "crps", "sharpness"]
    L = ["# Phase 9 — scores (frozen configuration)", "",
         f"Frozen: backtest `{FZ.backtest_tag}` settings, MOS `{name}`, sources {list(USE)}. "
         "Spread calibration and MOS fitted on 2022-2023; conformal, blend weights and abstain rule on 2024. "
         "Test A = 2025, Test B = 2026-01 to " + TEST_LAST + ". Standard errors: block bootstrap by LST day.", ""]
    for k, lab in (("A", "Test A (2025)"), ("B", f"Test B (2026-01..{TEST_LAST})")):
        L += [f"## {lab} — global", "", md(summ[k], cols), ""]
    L += ["## 2024 (frozen pipeline, for reference; conformal/blend/abstain in-sample)", "", md(s24, cols), ""]
    for k in periods:
        L += [f"## Stratified, Test {k} (floor / baseline / raw / blend log-loss)", ""]
        st = strat[k][strat[k].model.isin(["floor", "baseline", "raw", "blend"])]
        piv = st.pivot_table(index=["by", "value", "side"], columns="model", values="logloss").reset_index()
        n = st[st.model == "blend"].set_index(["by", "value", "side"]).n
        piv["n"] = piv.set_index(["by", "value", "side"]).index.map(n)
        piv["blend_hit"] = piv.set_index(["by", "value", "side"]).index.map(
            st[st.model == "blend"].set_index(["by", "value", "side"]).hit)
        L += [piv[["by", "value", "side", "n", "floor", "baseline", "raw", "blend", "blend_hit"]].to_markdown(index=False, floatfmt=".3f"), ""]
    L += ["## Abstain rule (fitted on 2024; `params/abstain.json`)", "",
          "Logistic P(mode correct | edge, mode probability, ensemble spread, day/night, season, sky class, sector); "
          "threshold = lowest with >= 85% hit on 2024 acted-on origins and >= 1% acted on.", "",
          "| Period | Side | Acted | Share | Hit among acted | 95% CI | Hit, all origins |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for lab, a in (("2024", ab24), ("Test A", ab["A"]), ("Test B", ab["B"])):
        for _, r in a.iterrows():
            L.append(f"| {lab} | {r.side} | {r.acted} | {r.share_acted:.3f} | {r.hit_acted:.3f} | {r.ci_lo:.3f}-{r.ci_hi:.3f} | {r.hit_all:.3f} |")
    L += ["", "## Reliability (final model)", ""]
    for k in periods:
        L += [f"Test {k}", "", rel[k].to_markdown(index=False, floatfmt=".3f"), ""]
    L += ["## Rank histogram of the realised extreme in the raw ensemble (deciles)", ""]
    for k in periods:
        L += [f"Test {k}", "", rh[k].to_markdown(index=False), ""]
    fl = stab[stab.flag]
    L += ["## Stability: Test B vs Test A (|diff| > 2 SE flagged)", "",
          f"{len(fl)} of {len(stab)} score cells flagged; global rows:", "",
          stab[stab.by == "all"].to_markdown(index=False, floatfmt=".3f"), "",
          "Flagged stratified cells (final model):", "",
          fl[(fl.by != "all") & (fl.model == "blend")].to_markdown(index=False, floatfmt=".3f"), ""]
    L += ["## 50 worst Test A windows (`out/phase9/worst/`)", "", worst["class"].value_counts().to_markdown(), ""]
    out(9, "scores.md").write_text("\n".join(L))
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    {"prep": prep, "backtest": run_backtest, "score": score}[sys.argv[1]]()
