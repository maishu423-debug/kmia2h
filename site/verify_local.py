"""Local verification (BUILD_SITE.md): /api/kalshi/algorithm/state carries our three models in the expected shape,
then a Playwright screenshot of the page to out/site/screenshot.png.

  site/.tools/venv/bin/python site/verify_local.py [frontend_url] [backend_url]

Needs the backend (gunicorn, RUN_BACKGROUND_TASKS=true) and the frontend (next start) running, and one completed
forecast cycle. Prints a JSON report; exits non-zero if a check fails."""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3000"
BACKEND = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
MODELS = ["kmia2h_high_2h", "kmia2h_low_2h", "kmia2h_day_high"]
KEYS = ["T0", "dist", "mean", "std", "mode", "mode_prob", "p10", "p90"]       # algorithm/state's per-model summary


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def check_state(state):
    f = state["forecast"]
    out = {"last_run_at": f["last_run_at"], "model_order": f.get("model_order"), "models": {}, "problems": []}
    if sorted(f["models"]) != sorted(MODELS):
        out["problems"].append(f"models are {sorted(f['models'])}")
    for name in MODELS:
        m = f["models"].get(name, {})
        if "error" in m:
            out["models"][name] = {"error": m["error"]}
            continue
        missing = [k for k in KEYS if m.get(k) is None]
        total = sum(d["prob"] for d in m.get("dist", []))
        out["models"][name] = {k: m.get(k) for k in ("T0", "mean", "std", "mode", "mode_prob", "p10", "p90")}
        out["models"][name]["n_bins"], out["models"][name]["prob_sum"] = len(m.get("dist", [])), round(total, 4)
        for k in ("action", "complete", "covers_to_utc"):
            if k in m:
                out["models"][name][k] = m[k]
        if missing or abs(total - 1) > 0.01:
            out["problems"].append(f"{name}: missing {missing}, prob sum {total:.3f}")
    ens = f.get("ensemble") or {}
    out["daily_high_for_trading"] = {k: ens.get(k) for k in ("mean", "mode", "mode_prob", "p10", "p90")}
    out["recorded_high_f"] = state["temp"]["daily_high_f"]
    out["recommendation"] = {"status": state["recommendation"]["status"], "reason": state["recommendation"]["reason"]}
    out["time_to_cutoff"] = state["time_to_cutoff"]["display"]
    return out


def screenshot(path):
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / "site" / ".tools" / "browsers"))
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(FRONTEND, wait_until="networkidle", timeout=90_000)
        page.wait_for_selector(".model-card", timeout=60_000)
        page.wait_for_function("document.querySelectorAll('.model-card .model-main').length >= 3", timeout=60_000)
        cards = page.locator(".model-grid .model-name").all_inner_texts()
        page.screenshot(path=str(path), full_page=True)
        browser.close()
    return cards


if __name__ == "__main__":
    report = check_state(get(f"{FRONTEND}/api/kalshi/algorithm/state"))
    report["backend_direct_equal_shape"] = sorted(get(f"{BACKEND}/api/kalshi/algorithm/state")["forecast"]["models"]) == sorted(MODELS)
    shot = ROOT / "out" / "site" / "screenshot.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    report["page_model_cards"] = screenshot(shot)
    report["screenshot"] = str(shot.relative_to(ROOT))
    print(json.dumps(report, indent=1, default=str))
    sys.exit(1 if report["problems"] else 0)
