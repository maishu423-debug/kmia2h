"""Phase 5: the filter on systems whose truth is known."""
import copy

import numpy as np
import pytest
from scipy import stats

import helpers
from kmia2h.filter import enkf, obs_operators as oo, process_model as pm
from kmia2h.filter.enkf import EnKF
from kmia2h.filter.obs_operators import Obs

RP = {"theta_f": 0.3, "sigma_f": 0.15, "theta_s": 0.04, "sigma_s": 0.05}


def linear_setup(cfg, n):
    """One-dimensional state with a known Gaussian posterior."""
    c = copy.deepcopy(cfg)
    c["filter"]["inflation"] = 1.0
    c["filter"]["obs_sigma"]["goes"] = 0.5
    x0 = 10.0 + 2.0 * np.random.default_rng(3).standard_normal((n, 1))
    return c, x0


def test_analysis_matches_the_kalman_solution(cfg):
    c, x0 = linear_setup(cfg, 20000)
    kf = EnKF(x0, lambda x, k, rng: x, c, seed=1, operators={"goes": lambda x: x[:, 0]},
              path_fn=lambda x: x[:, 0], constrain_fn=lambda x: x, inflated=[0])
    m0, v0 = x0.mean(), x0.var(ddof=1)
    kf.assimilate([Obs(0, 0, "goes", 12.0)])
    gain = v0 / (v0 + 0.25)
    assert kf.state.mean() == pytest.approx(m0 + gain * (12.0 - m0), abs=0.02)
    assert kf.state.var(ddof=1) == pytest.approx((1 - gain) * v0, rel=0.05)


def test_inflation_multiplies_the_spread_once_per_analysis(cfg):
    c, x0 = linear_setup(cfg, 2000)
    c["filter"]["inflation"] = 1.5
    c["filter"]["obs_sigma"]["goes"] = 1e6       # the observation carries no weight
    kf = EnKF(x0, lambda x, k, rng: x, c, seed=1, operators={"goes": lambda x: x[:, 0]},
              path_fn=lambda x: x[:, 0], constrain_fn=lambda x: x, inflated=[0])
    kf.assimilate([Obs(0, 0, "goes", 12.0), Obs(0, 0, "goes", 12.0)])
    assert kf.state.std(ddof=1) == pytest.approx(1.5 * x0.std(ddof=1), rel=1e-3)


# --- Process model ------------------------------------------------------------------------------

def base_forcing(n):
    t = np.arange(n, dtype=float)
    day = 1440.0
    cloud_blocks = np.random.default_rng(11).uniform(0.0, 1.0, n // 180 + 2)
    return {
        "s_clr": 900.0 * np.maximum(np.sin(2 * np.pi * (t - 360) / day), 0.0),
        "c": cloud_blocks[(t // 180).astype(int)],
        "t_adv": 76.0 + 5.0 * np.sin(2 * np.pi * t / (37 * 60.0)),
        "t_sky": 52.0 + 9.0 * np.sin(2 * np.pi * t / (17 * 60.0)),
        "r": np.zeros(n),
    }


def regime(n, rp=RP):
    return {k: np.full(n, v) for k, v in rp.items()}


def quiet(cfg):
    c = copy.deepcopy(cfg)
    c["filter"].update({"c_sigma": 0.0, "t_sigma": 0.0, "t_adv_sigma": 0.0})
    c["filter"]["coef_rw_sigma"] = {k: 0.0 for k in c["filter"]["coef_rw_sigma"]}
    return c


def test_ou_step_has_the_exact_stationary_variance():
    rng = np.random.default_rng(0)
    x = np.zeros(400000)
    theta, sigma = 0.5, 0.4     # Euler-Maruyama at dt = 1 would be 33% too wide here
    for _ in range(40):
        x = pm.ou_step(x, theta, sigma, rng.standard_normal(x.size))
    assert x.var() == pytest.approx(sigma ** 2 / (2 * theta), rel=0.01)


def test_lag_states_shift_and_hold_sensor_temperature(cfg):
    c = quiet(cfg)
    x = pm.init_ensemble(5, 80.0, 0.5, c, np.random.default_rng(0))
    y = [pm.sensor_temperature(x)]
    step = pm.make_step(base_forcing(20), regime(20), c)
    rng = np.random.default_rng(1)
    for k in range(6):
        x = step(x, k, rng)
        y.append(pm.sensor_temperature(x))
    for i, name in enumerate(("T_m1", "T_m2", "T_m3", "T_m4"), start=1):
        assert x[:, pm.IDX[name]] == pytest.approx(y[-1 - i])


def test_constraints_clip_coefficients_and_cloud(cfg):
    x = np.zeros((2, len(pm.STATE)))
    x[:, pm.IDX["kappa"]] = [1.7, -0.2]
    x[:, pm.IDX["b"]] = [-1.0, 5.0]
    x[:, pm.IDX["C_eff"]] = [1.5, -0.5]
    pm.constrain(x, cfg)
    assert list(x[:, pm.IDX["kappa"]]) == [1.0, 0.0]
    assert list(x[:, pm.IDX["b"]]) == [0.0, cfg["physics"]["bounds"]["b"][1]]
    assert list(x[:, pm.IDX["C_eff"]]) == [1.0, 0.0]


def test_regime_lookup_uses_the_finest_key_available():
    table = {"wet|True|sct|5to10": {"id": 1}, "wet|True|sct": {"id": 2}, "wet|True": {"id": 3}}
    assert pm.lookup_regime(table, "wet", True, "sct", "5to10")["id"] == 1
    assert pm.lookup_regime(table, "wet", True, "sct", "gt15")["id"] == 2
    assert pm.lookup_regime(table, "wet", True, "ovc", "gt15")["id"] == 3
    with pytest.raises(KeyError):
        pm.lookup_regime(table, "dry", True, "ovc", "gt15")


def test_regime_switch_does_not_reinitialise_the_ensemble(cfg):
    c = quiet(cfg)
    n = 10
    rp = {k: np.r_[np.full(5, v), np.full(5, 2 * v)] for k, v in RP.items()}
    x0 = pm.init_ensemble(50, 80.0, 0.5, c, np.random.default_rng(0))
    step = pm.make_step(base_forcing(n), rp, c)
    a = step(x0, 3, np.random.default_rng(1))
    b = step(a, 4, np.random.default_rng(2))
    assert np.corrcoef(a[:, pm.IDX["T"]], b[:, pm.IDX["T"]])[0, 1] > 0.99


# --- Observation operators and timing ------------------------------------------------------------

def test_five_minute_mean_uses_the_five_latest_values():
    x = np.zeros((1, len(pm.STATE)))
    x[0, pm.IDX["T"]] = 79.0
    x[0, pm.IDX["X_slow"]] = 0.5
    x[0, pm.IDX["X_fast"]] = 0.5
    for name, value in zip(("T_m1", "T_m2", "T_m3", "T_m4"), (81.0, 82.0, 83.0, 84.0)):
        x[0, pm.IDX[name]] = value
    assert oo.five_minute_mean(x)[0] == pytest.approx((80 + 81 + 82 + 83 + 84) / 5)


def test_hfmetar_chain_matches_a_worked_example():
    # 80.4 -> 80 degF -> 26.67 -> 27 degC -> 80.6 degF
    assert oo.hfmetar_chain(80.4, helpers, "half_up") == pytest.approx(80.6)


def test_hfmetar_rounding_error_is_near_the_configured_sigma(cfg):
    # The spec's 0.52 counts the whole-degC step only. The whole-degF step before it adds 1/12 degF^2.
    t = np.random.default_rng(0).uniform(70, 90, 200000)
    err = oo.hfmetar_chain(t, helpers, "half_up") - t
    assert err.std() == pytest.approx(cfg["filter"]["obs_sigma"]["hfmetar"] * np.sqrt(1 + (1 / 3.24)), rel=0.1)


def make_filter(cfg, n_total, members=60, seed=5):
    x0 = pm.init_ensemble(members, 80.0, 0.5, cfg, np.random.default_rng(seed))
    return EnKF(x0, pm.make_step(base_forcing(n_total), regime(n_total), cfg), cfg, seed=seed)


def test_future_observation_is_rejected(cfg):
    kf = make_filter(cfg, 30)
    for _ in range(5):
        kf.advance()
    with pytest.raises(ValueError, match="look-ahead"):
        kf.assimilate([Obs(valid_k=5, available_k=13, kind="hfmetar", value=80.0)])


def test_late_observation_equals_on_time_assimilation(cfg):
    a, b = make_filter(cfg, 80), make_filter(cfg, 80)
    for _ in range(20):
        a.advance()
    a.assimilate([Obs(20, 20, "hfmetar", 82.0)])
    for _ in range(8):
        a.advance()
    for _ in range(28):
        b.advance()
    b.assimilate([Obs(20, 28, "hfmetar", 82.0)])
    assert np.array_equal(a.state, b.state)


def test_observations_older_than_the_lag_window_are_dropped(cfg):
    kf = make_filter(cfg, 80)
    for _ in range(40):
        kf.advance()
    before = kf.state.copy()
    kf.assimilate([Obs(2, 30, "hfmetar", 82.0)])
    assert kf.dropped == 1 and np.array_equal(kf.state, before)


def test_snapshot_has_the_ensemble_and_a_smoothed_two_hour_path(cfg):
    kf = make_filter(cfg, 400, members=40)
    for _ in range(300):
        kf.advance()
    kf.assimilate([Obs(295, 300, "hfmetar", 86.0)])
    snap = kf.snapshot()
    assert snap["ensemble"].shape == (40, len(pm.STATE))
    assert snap["paths"].shape == (40, cfg["filter"]["path_window_min"])
    assert snap["paths"][:, -1] == pytest.approx(pm.sensor_temperature(snap["ensemble"]))


def test_smoother_moves_the_path_toward_a_late_observation(cfg):
    kf = make_filter(cfg, 200, members=200)
    for _ in range(100):
        kf.advance()
    before = kf.snapshot()["paths"].mean(axis=0)
    kf.assimilate([Obs(100, 100, "hfmetar", before[-1] + 5.0)])
    after = kf.snapshot()["paths"].mean(axis=0)
    assert after[-1] > before[-1] + 1.0
    assert after[-6:].mean() - before[-6:].mean() > 1.0


# --- Recovery of a simulated truth ----------------------------------------------------------------

def simulate_truth(cfg, n, coef, seed):
    """Truth from the process model with fixed coefficients and residual noise."""
    x = np.zeros((1, len(pm.STATE)))
    x[0, pm.IDX["T"]] = 78.0
    for k, v in coef.items():
        x[0, pm.IDX[k]] = v
    x[0, pm.IDX["C_eff"]] = 0.5
    for name in ("T_m1", "T_m2", "T_m3", "T_m4"):
        x[0, pm.IDX[name]] = 78.0
    forcing = base_forcing(n)
    step = pm.make_step(forcing, regime(n), cfg)
    rng = np.random.default_rng(seed)
    out = np.zeros((n, x.shape[1]))
    out[0] = x[0]
    for k in range(n - 1):
        x = step(x, k, rng)
        out[k + 1] = x[0]
    return out


def observations(truth, cfg, seed, rounded=False):
    """HFMETAR every 5 min (8 min late), hourly METAR and a GOES cloud value every 10 min."""
    rng = np.random.default_rng(seed)
    obs, f = [], cfg["filter"]["obs_sigma"]
    mean5 = oo.five_minute_mean(truth)
    for k in range(5, len(truth)):
        if k % 5 == 0:
            y = (oo.hfmetar_chain(mean5[k], helpers, "half_up") if rounded
                 else mean5[k] + rng.normal(0, f["hfmetar"]))
            obs.append(Obs(k, k + 8, "hfmetar", float(y)))
        if k % 60 == 53:
            obs.append(Obs(k, k + 5, "metar", float(mean5[k] + rng.normal(0, f["metar"]))))
        if k % 10 == 0:
            obs.append(Obs(k, k + 10, "goes", float(np.clip(truth[k, pm.IDX["C_eff"]] + rng.normal(0, f["goes"]), 0, 1))))
    return obs


TRUE_COEF = {"a": 8.0e-5, "b": 0.015, "c": 4.0e-4, "kappa": 0.55}


def recovery_cfg(cfg):
    c = quiet(cfg)
    c["filter"]["c_sigma"] = 0.005
    c["filter"]["t_sigma"] = 0.005
    c["filter"]["coef_rw_sigma"] = {"a": 3e-8, "b": 5e-6, "c": 1.5e-7, "kappa": 2e-4}
    return c


@pytest.fixture(scope="module")
def recovery():
    cfg = recovery_cfg(helpers.load_cfg())
    n = 20 * 1440
    truth_cfg = quiet(cfg)
    truth = simulate_truth(truth_cfg, n, TRUE_COEF, seed=21)
    x0 = pm.init_ensemble(200, 78.0, 0.5, cfg, np.random.default_rng(22))
    kf = EnKF(x0, pm.make_step(base_forcing(n), regime(n), cfg), cfg, seed=23)
    obs = observations(truth, cfg, seed=24)
    every = list(range(2000, n - 1, 30))
    snaps = enkf.run(kf, obs, n - 1, snapshot_at=every)
    return {"cfg": cfg, "truth": truth, "kf": kf, "snaps": snaps, "every": every, "n": n, "obs": obs}


def test_filter_recovers_the_true_coefficients(recovery):
    late = [recovery["snaps"][k]["ensemble"] for k in recovery["every"][-288:]]
    est = np.concatenate(late).mean(axis=0)
    for name, true in TRUE_COEF.items():
        assert est[pm.IDX[name]] == pytest.approx(true, rel=0.15), name


def test_filter_moves_coefficients_from_the_prior_toward_truth(recovery):
    prior = recovery["cfg"]["physics"]["prior"]
    late = [recovery["snaps"][k]["ensemble"] for k in recovery["every"][-288:]]
    est = np.concatenate(late).mean(axis=0)
    improved = [abs(est[pm.IDX[k]] - TRUE_COEF[k]) < abs(prior[k] - TRUE_COEF[k]) for k in TRUE_COEF]
    assert all(improved), dict(zip(TRUE_COEF, improved))


def test_filter_recovers_the_state(recovery):
    truth, snaps, every = recovery["truth"], recovery["snaps"], recovery["every"]
    y_true = pm.sensor_temperature(truth[every])
    y_est = np.array([pm.sensor_temperature(snaps[k]["ensemble"]).mean() for k in every])
    hf = sorted((o.available_k, o.value) for o in recovery["obs"] if o.kind == "hfmetar")
    avail = np.array([a for a, _ in hf])
    last_obs = np.array([v for _, v in hf])[np.searchsorted(avail, every, side="right") - 1]
    rmse = np.sqrt(np.mean((y_est - y_true) ** 2))
    assert rmse < 0.5
    assert rmse < np.sqrt(np.mean((last_obs - y_true) ** 2))    # beats the last HFMETAR alone
    c_true, c_est = truth[every, pm.IDX["C_eff"]], np.array([snaps[k]["ensemble"][:, pm.IDX["C_eff"]].mean() for k in every])
    assert np.sqrt(np.mean((c_est - c_true) ** 2)) < 0.12


def test_innovations_are_calibrated(recovery):
    d = recovery["kf"].innovations()
    z = d.loc[d["kind"] == "hfmetar", "normalised"].to_numpy()[2000:]
    assert abs(z.mean()) < 0.1
    assert 0.8 < z.var() < 1.2
    assert abs(np.corrcoef(z[:-1], z[1:])[0, 1]) < 0.2


def test_rank_of_truth_in_the_analysis_ensemble_is_roughly_uniform(recovery):
    truth, snaps, every = recovery["truth"], recovery["snaps"], recovery["every"]
    y_true = pm.sensor_temperature(truth[every])
    ranks = np.array([(pm.sensor_temperature(snaps[k]["ensemble"]) < t).mean() for k, t in zip(every, y_true)])
    counts = np.histogram(ranks, bins=10, range=(0, 1))[0]
    # Snapshots 30 minutes apart are correlated, so the test uses an effective sample of 1 in 4.
    assert stats.chisquare(counts / 4.0).pvalue > 0.01, counts


def test_recovery_survives_real_rounding_of_the_observation(recovery):
    cfg = recovery["cfg"]
    n = 8 * 1440
    truth = simulate_truth(quiet(cfg), n, TRUE_COEF, seed=31)
    x0 = pm.init_ensemble(100, 78.0, 0.5, cfg, np.random.default_rng(32))
    kf = EnKF(x0, pm.make_step(base_forcing(n), regime(n), cfg), cfg, seed=33)
    every = list(range(2000, n - 1, 30))
    snaps = enkf.run(kf, observations(truth, cfg, seed=34, rounded=True), n - 1, snapshot_at=every)
    y_true = pm.sensor_temperature(truth[every])
    y_est = np.array([pm.sensor_temperature(snaps[k]["ensemble"]).mean() for k in every])
    assert np.sqrt(np.mean((y_est - y_true) ** 2)) < 0.6
    assert abs(np.mean(y_est - y_true)) < 0.2
