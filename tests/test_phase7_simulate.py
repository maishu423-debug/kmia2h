"""Phase 7: Monte Carlo bin probabilities against cases with an analytic answer."""
import itertools
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import helpers
from kmia2h import interfaces, simulate
from kmia2h.filter import process_model as pm
from kmia2h.physics import ode

ORIGIN = pd.Timestamp("2024-07-01 15:00", tz="UTC")
NO_NOISE = {"theta_f": 0.3, "sigma_f": 0.0, "theta_s": 0.03, "sigma_s": 0.0, "white": 0.0}


def forcing(n=120, t_adv=80.0, **kw):
    f = {"s_clr": np.zeros(n), "c": np.zeros(n), "t_adv": np.full(n, float(t_adv)), "t_sky": np.full(n, 60.0), "r": np.zeros(n)}
    f.update({k: np.broadcast_to(v, (n,)).astype(float) for k, v in kw.items()})
    return f


def ensemble(t, members=None, cfg_coefs=None):
    """Frozen-physics ensemble: no ODE tendency, lag states equal to the member's temperature."""
    t = np.atleast_1d(np.asarray(t, float))
    x = np.zeros((len(t), len(pm.STATE)))
    x[:, pm.IDX["T"]] = t
    for name in ("T_m1", "T_m2", "T_m3", "T_m4"):
        x[:, pm.IDX[name]] = t
    for k, v in (cfg_coefs or {}).items():
        x[:, pm.IDX[k]] = v
    return {"ensemble": x}


def frozen(cfg):
    """Config with no jitter, no forcing perturbation and zero cloud effect."""
    c = {**cfg, "simulate": {**cfg["simulate"], "jitter_frac": 0.0, "cloud_sigma": 0.0}}
    return c


def probs(table):
    return {int(k): v for k, v in table.items()}


def test_helpers_satisfy_the_target_protocol():
    assert isinstance(helpers, interfaces.Target)
    assert isinstance(helpers.StubWindow(), interfaces.Window)


def test_settle_copy_matches_the_spec_cases():
    assert helpers.settle(np.full(124, 80.0), "half_up") == (80.0, 80.0)
    assert helpers.reported(np.array([80, 80, 80, 81, 81.0]), "half_up")[-1] == 80.0
    t = np.array([80, np.nan, 80, 81, 81.0])   # four valid, mean exactly 80.5
    assert helpers.reported(t, "half_up")[-1] == 81.0
    assert np.isnan(helpers.reported(np.array([80, np.nan, np.nan, 81, 81.0]), "half_up")[-1])


def test_constant_state_settles_with_certainty(cfg):
    r = simulate.run(ensemble([80.3]), forcing(), NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=500)
    assert r.forecast["p_high"] == {"80": 1.0} and r.forecast["p_low"] == {"80": 1.0}
    assert r.forecast["p_joint"] == {"80|80": 1.0}


def test_gaussian_initial_state_matches_the_normal_cdf(cfg):
    mu, sigma = 80.3, 0.8
    t = stats.norm(mu, sigma).rvs(300000, random_state=1)
    r = simulate.run(ensemble(t), forcing(n=10), NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=300000)
    p = probs(r.forecast["p_high"])
    for k in range(78, 83):
        exact = stats.norm(mu, sigma).cdf(k + 0.5) - stats.norm(mu, sigma).cdf(k - 0.5)
        assert p.get(k, 0.0) == pytest.approx(exact, abs=0.004)
    assert r.forecast["p_high"] == r.forecast["p_low"]


def test_two_member_ensemble_gives_the_exact_split(cfg):
    t = np.r_[np.full(700, 80.2), np.full(300, 83.1)]
    r = simulate.run(ensemble(t), forcing(n=10), NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=200000)
    p = probs(r.forecast["p_high"])
    assert set(p) == {80, 83} and p[80] == pytest.approx(0.7, abs=0.005)


def white_pmf(mu, sigma):
    ks = np.arange(int(mu - 8 * sigma - 2), int(mu + 8 * sigma + 3))
    d = stats.norm(mu, sigma)
    return ks, d.cdf(ks + 0.5) - d.cdf(ks - 0.5)


def test_one_minute_window_matches_exact_rolling_arithmetic(cfg):
    """Lead minutes are all 80, so reported(t+1) = round((320 + W) / 5), W = rounded Gaussian."""
    mu, sigma = 80.3, 6.0
    rp = {**NO_NOISE, "white": sigma ** 2}
    r = simulate.run(ensemble([mu - 0.3] * 3), forcing(n=1, t_adv=mu), {**rp}, helpers, frozen(cfg), ORIGIN, n_paths=400000)

    # The ODE is frozen at 80.0 so the sensor mean is 80.0, not mu.
    ks, pm_w = white_pmf(80.0, sigma)
    exact = defaultdict(float)
    for w, p in zip(ks, pm_w):
        exact[int(np.floor((320 + w) / 5 + 0.5))] += p
    got = probs(r.forecast["p_high"])
    for k, v in exact.items():
        assert got.get(k, 0.0) == pytest.approx(v, abs=0.005)


def test_two_minute_joint_table_matches_exact_enumeration(cfg):
    sigma = 5.0
    rp = {**NO_NOISE, "white": sigma ** 2}
    r = simulate.run(ensemble([80.0] * 3), forcing(n=2), rp, helpers, frozen(cfg), ORIGIN, n_paths=400000)
    ks, p = white_pmf(80.0, sigma)
    exact_joint, exact_high = defaultdict(float), defaultdict(float)
    for (w1, p1), (w2, p2) in itertools.product(zip(ks, p), zip(ks, p)):
        r1, r2 = np.floor((320 + w1) / 5 + 0.5), np.floor((240 + w1 + w2) / 5 + 0.5)
        hi, lo = int(max(r1, r2)), int(min(r1, r2))
        exact_joint[f"{hi}|{lo}"] += p1 * p2
        exact_high[hi] += p1 * p2
    got = r.forecast["p_joint"]
    for key, v in exact_joint.items():
        assert got.get(key, 0.0) == pytest.approx(v, abs=0.006)
    high = probs(r.forecast["p_high"])
    for k, v in exact_high.items():
        assert high.get(k, 0.0) == pytest.approx(v, abs=0.006)


def test_deterministic_relaxation_settles_on_the_analytic_path(cfg):
    b = 0.02
    n = 120
    t0, t_adv = 75.0, 85.0
    coefs = {"b": b}
    r = simulate.run(ensemble([t0], cfg_coefs=coefs), forcing(n, t_adv=t_adv), NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=50)
    euler = t_adv + (t0 - t_adv) * (1 - b) ** np.arange(1, n + 1)
    series = helpers.round_rule(np.r_[np.full(4, t0), euler], "half_up")
    hi, lo = helpers.settle(series, "half_up")
    assert probs(r.forecast["p_high"]) == {int(hi): 1.0}
    assert probs(r.forecast["p_low"]) == {int(lo): 1.0}
    assert r.forecast["mean_path"][-1] == pytest.approx(euler[-1], abs=1e-9)


def test_mean_path_matches_the_rk45_integration_closely(cfg):
    x = ensemble([75.0], cfg_coefs={"b": 0.02})
    r = simulate.run(x, forcing(), NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=20)
    exact = ode.integrate(75.0, {"a": 0.0, "b": 0.02, "c": 0.0, "kappa": 0.0, "lam": 0.0}, forcing(), cfg)["t"].to_numpy()
    assert np.max(np.abs(np.array(r.forecast["mean_path"]) - exact)) < 0.06


def test_jitter_adds_the_configured_variance(cfg):
    sigma = 0.8
    t = stats.norm(80.3, sigma).rvs(100000, random_state=5)
    c = {**frozen(cfg), "simulate": {**frozen(cfg)["simulate"], "jitter_frac": 0.5}}
    r = simulate.run(ensemble(t), forcing(n=3), NO_NOISE, helpers, c, ORIGIN, n_paths=200000)
    assert r.paths[:, 4].std() == pytest.approx(np.sqrt(sigma ** 2 * 1.25 + 1 / 12), rel=0.02)


def test_analysis_residual_seeds_the_noise_paths(cfg):
    x = ensemble([80.0])
    x["ensemble"][:, pm.IDX["X_slow"]] = 3.0
    r = simulate.run(x, forcing(n=2), {**NO_NOISE, "theta_s": 0.0}, helpers, frozen(cfg), ORIGIN, n_paths=20)
    assert r.paths[:, 4:].mean() == pytest.approx(83.0)
    # The lead minute at the origin carries the residual too.
    assert r.paths[:, 3].mean() == pytest.approx(83.0)


def test_forcing_perturbation_widens_only_when_it_should(cfg):
    x = ensemble([85.0] * 200, cfg_coefs={"a": 1e-3, "kappa": 1.0})
    x["ensemble"][:, pm.IDX["C_eff"]] = 0.5
    f = forcing(s_clr=800.0, c=0.5)
    narrow = simulate.run(x, f, NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=1000)
    wide_cfg = {**frozen(cfg), "simulate": {**frozen(cfg)["simulate"], "cloud_sigma": 0.2}}
    wide = simulate.run(x, f, NO_NOISE, helpers, wide_cfg, ORIGIN, n_paths=1000)
    assert narrow.high.std() < wide.high.std()
    assert narrow.high.std() == pytest.approx(0.0)


def test_cloud_paths_respond_to_the_motion_vector(cfg):
    c = cfg["physics"]["cloud"]
    f = np.zeros((c["n_sectors"], c["n_rings"]))
    f[0, 1] = 1.0
    cloud = {"cf_disc": 0.0, "cf_field": f, "u": 0.0, "v": -8.0}
    base = np.zeros(120)
    no_cloud_sigma = {**cfg, "simulate": {**cfg["simulate"], "cloud_sigma": 0.0}}
    paths = simulate.cloud_paths(base, 500, no_cloud_sigma, np.random.default_rng(0), cloud)
    assert paths.max(axis=0).max() > 0 and paths[:, 20:40].std(axis=0).max() > 0.02
    assert (paths >= 0).all() and (paths <= 1).all()


def test_rain_cell_lowers_the_low(cfg):
    x = ensemble([80.0] * 10, cfg_coefs={"b": cfg["physics"]["prior"]["b"]})
    cells = pd.DataFrame({"arrival_min": [30.0], "drop_f": [5.0]})
    r = simulate.run(x, forcing(), NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=2000, rain_cells=cells)
    assert min(probs(r.forecast["p_low"])) <= 77
    assert max(probs(r.forecast["p_high"])) == 80


def test_elapsed_window_high_pins_the_distribution(cfg):
    x = ensemble([80.0] * 5)
    elapsed = 30
    past = np.full((5, 120), 80.0)
    past[:, -15:-8] = 90.0
    x["paths"] = past
    r = simulate.run(x, forcing(n=90), NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=100, elapsed_min=elapsed)
    series = np.r_[past[0, -(elapsed + 4):], np.full(90, 80.0)]
    hi, lo = helpers.settle(series, "half_up")
    assert hi == 90.0 and probs(r.forecast["p_high"]) == {90: 1.0}
    assert probs(r.forecast["p_low"]) == {int(lo): 1.0}


def test_elapsed_window_needs_enough_history(cfg):
    x = ensemble([80.0] * 5)
    x["paths"] = np.full((5, 10), 80.0)
    with pytest.raises(ValueError, match="elapsed"):
        simulate.run(x, forcing(n=90), NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=10, elapsed_min=30)


def test_stub_window_drives_an_elapsed_run(cfg):
    w = helpers.StubWindow(elapsed=10, remaining=110)
    x = ensemble([80.0] * 5)
    x["paths"] = np.full((5, 120), 80.0)
    r = simulate.run(x, forcing(n=w.remaining_min(ORIGIN)), NO_NOISE, helpers, frozen(cfg), ORIGIN, n_paths=10,
                     elapsed_min=w.elapsed_min(ORIGIN))
    assert r.paths.shape == (10, 4 + 10 + 110)
    assert r.forecast["window_end_utc"] == (ORIGIN + pd.Timedelta(minutes=110)).isoformat()


# --- Output contract --------------------------------------------------------------------------------

def noisy_run(cfg, seed=1, n_paths=2000):
    rng = np.random.default_rng(0)
    x = pm.init_ensemble(300, 80.0, 0.5, cfg, rng)
    rp = {"theta_f": 0.3, "sigma_f": 0.2, "theta_s": 0.03, "sigma_s": 0.05, "white": 0.05, "lam_per_hour": 1.0,
          "p_drop": 0.7, "mean_drop": 0.8, "mean_rise": 0.5, "jump_min": 1.0, "nu": 6.0}
    return simulate.run({"ensemble": x}, forcing(s_clr=700.0, c=0.4), rp, helpers, cfg, ORIGIN, seed=seed, n_paths=n_paths,
                        regime="day-onshore-sct", latest_obs=80.0)


def test_forecast_matches_the_schema_and_sums_to_one(cfg):
    r = noisy_run(cfg)
    assert simulate.validate_forecast(r.forecast) == []
    for table in ("p_high", "p_low", "p_joint"):
        assert sum(r.forecast[table].values()) == pytest.approx(1.0, abs=1e-9)
    assert len(r.forecast["mean_path"]) == 120
    json.dumps(r.forecast)


def test_validate_forecast_reports_problems():
    assert simulate.validate_forecast({"p_high": {}}) != []
    bad = {k: 0 for k in ("origin_utc", "window_end_utc", "n_paths", "edge_high", "edge_low", "regime", "mean_path")}
    bad.update({"p_high": {"80": 0.5}, "p_low": {"80": 1.0}, "p_joint": {"79|80": 1.0}, "q_max": {}, "q_min": {}})
    problems = simulate.validate_forecast(bad)
    assert any("p_high sums" in p for p in problems) and any("high below low" in p for p in problems)


def test_marginals_agree_with_the_joint(cfg):
    fc = noisy_run(cfg).forecast
    marginal = defaultdict(float)
    for key, v in fc["p_joint"].items():
        marginal[key.split("|")[0]] += v
    for k, v in fc["p_high"].items():
        assert marginal[k] == pytest.approx(v, abs=1e-12)


def test_same_seed_is_identical_and_a_new_seed_differs(cfg):
    a, b, c = noisy_run(cfg, seed=7), noisy_run(cfg, seed=7), noisy_run(cfg, seed=8)
    assert a.forecast == b.forecast
    assert a.forecast["p_high"] != c.forecast["p_high"]


def test_paths_are_whole_numbers_and_quantiles_ordered(cfg):
    r = noisy_run(cfg)
    assert np.array_equal(r.paths, np.round(r.paths))
    q = list(r.forecast["q_max"].values())
    assert q == sorted(q) and set(r.forecast["q_max"]) == {"05", "25", "50", "75", "95"}


def test_edge_is_computed_from_the_mean_path_rolling_extreme(cfg):
    r = simulate.run(ensemble([80.3]), forcing(n=10, t_adv=80.3), {**NO_NOISE}, helpers, frozen(cfg), ORIGIN, n_paths=5)
    assert r.forecast["edge_high"] == pytest.approx(abs(0.3 - 0.5))


def test_sanity_checks_pass_on_a_normal_run_and_flag_a_bad_one(cfg):
    assert noisy_run(cfg).checks["first_minute_near_obs"] is True
    x = pm.init_ensemble(100, 80.0, 0.5, cfg, np.random.default_rng(0))
    rp = {**NO_NOISE, "white": 100.0}
    r = simulate.run({"ensemble": x}, forcing(), rp, helpers, cfg, ORIGIN, n_paths=200, latest_obs=95.0, floor_range=(0.0, 0.01))
    assert r.checks["first_minute_near_obs"] is False
    assert r.checks["steps_within_limit"] is False
    assert r.checks["spread_in_floor_range"] is False


def test_4000_paths_run_in_under_five_seconds(cfg):
    start = time.perf_counter()
    noisy_run(cfg, n_paths=4000)
    assert time.perf_counter() - start < 5.0
