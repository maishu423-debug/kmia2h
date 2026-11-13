"""Phase 6: simulate with known parameters, fit, and compare."""
import numpy as np
import pytest

import helpers
from kmia2h.residual import residual_fit as rf, residual_sim as rs

TRUE = {"theta_f": 0.3, "sigma_f": 0.25, "theta_s": 0.035, "sigma_s": 0.07, "white": 0.04}


def series(rp, n_paths=500, n_steps=1000, seed=0):
    sim = rs.simulate(rp, n_paths, n_steps, np.random.default_rng(seed))
    total = sim["x_slow"] + sim["x_fast"] + sim["white"]
    return total, np.repeat(np.arange(n_paths), n_steps)


def test_ou_stationary_variance_matches_theory_within_5_percent():
    rp = {**TRUE, "white": 0.0}
    sim = rs.simulate(rp, 20000, 300, np.random.default_rng(1))
    assert sim["x_fast"][:, -1].var() == pytest.approx(TRUE["sigma_f"] ** 2 / (2 * TRUE["theta_f"]), rel=0.05)
    assert sim["x_slow"][:, -1].var() == pytest.approx(TRUE["sigma_s"] ** 2 / (2 * TRUE["theta_s"]), rel=0.05)


def test_ou_variance_is_right_from_a_cold_start_and_at_a_fast_rate():
    rp = {"theta_f": 0.5, "sigma_f": 0.4, "theta_s": 0.03, "sigma_s": 0.0, "white": 0.0}
    sim = rs.simulate(rp, 40000, 40, np.random.default_rng(2), init_slow=np.zeros(40000), init_fast=np.zeros(40000))
    assert sim["x_fast"][:, -1].var() == pytest.approx(0.4 ** 2 / 1.0, rel=0.05)


def test_ou_autocorrelation_matches_exp_minus_theta():
    rp = {"theta_f": 0.2, "sigma_f": 0.3, "theta_s": 0.03, "sigma_s": 0.0, "white": 0.0}
    x = rs.simulate(rp, 20000, 50, np.random.default_rng(3))["x_fast"]
    assert np.corrcoef(x[:, 20], x[:, 25])[0, 1] == pytest.approx(np.exp(-1.0), abs=0.02)


def test_two_timescales_and_white_noise_are_recovered(cfg):
    x, seg = series(TRUE, seed=4)
    c = rf.autocovariance(x.ravel(), cfg["residual"]["acf_maxlag"], seg)
    fit = rf.fit_two_exponentials(c, cfg)
    assert fit["theta_f"] == pytest.approx(TRUE["theta_f"], rel=0.15)
    assert fit["theta_s"] == pytest.approx(TRUE["theta_s"], rel=0.15)
    assert fit["A_f"] == pytest.approx(TRUE["sigma_f"] ** 2 / (2 * TRUE["theta_f"]), rel=0.15)
    assert fit["A_s"] == pytest.approx(TRUE["sigma_s"] ** 2 / (2 * TRUE["theta_s"]), rel=0.15)
    assert fit["white"] == pytest.approx(TRUE["white"], abs=0.015)


def test_sde_parameters_are_recovered_from_the_autocovariance(cfg):
    x, seg = series(TRUE, seed=5)
    p = rf.ou_params(rf.fit_two_exponentials(rf.autocovariance(x.ravel(), 90, seg), cfg), cfg)
    assert p["sigma_f"] == pytest.approx(TRUE["sigma_f"], rel=0.1)
    assert p["sigma_s"] == pytest.approx(TRUE["sigma_s"], rel=0.15)


def test_autocovariance_does_not_cross_segments():
    x = np.r_[np.full(50, 1.0), np.full(50, -1.0)]
    seg = np.r_[np.zeros(50), np.ones(50)]
    c = rf.autocovariance(x, 5, seg)
    assert c[5] == pytest.approx(1.0)
    assert rf.autocovariance(x, 5)[5] < 1.0


def test_autocovariance_skips_missing_values():
    x = np.sin(np.arange(2000) * 0.1)
    y = x.copy()
    y[::17] = np.nan
    assert rf.autocovariance(y, 3) == pytest.approx(rf.autocovariance(x, 3), abs=0.01)


def test_small_component_collapses_to_a_single_ou(cfg):
    row = {"A_f": 0.5, "theta_f": 0.2, "A_s": 0.02, "theta_s": 0.03, "white": 0.2}
    p = rf.ou_params(row, cfg)
    assert p["sigma_s"] == 0.0
    assert p["sigma_f"] ** 2 / (2 * p["theta_f"]) == pytest.approx(0.52)


def test_white_noise_excludes_the_quantisation_part(cfg):
    row = {"A_f": 0.3, "theta_f": 0.2, "A_s": 0.3, "theta_s": 0.03, "white": 0.2}
    assert rf.ou_params(row, cfg)["white"] == pytest.approx(0.2 - 1.0 / 12.0)
    assert rf.ou_params({**row, "white": 0.05}, cfg)["white"] == 0.0


def test_quantised_white_noise_is_not_simulated_twice(cfg):
    """After rounding, the variance of a constant-mean series is back to the Phase 3 white noise."""
    target_white = 0.25
    white = rf.ou_params({"A_f": 0.0, "theta_f": 0.2, "A_s": 0.0, "theta_s": 0.03, "white": target_white}, cfg)["white"]
    v = 80.3 + np.sqrt(white) * np.random.default_rng(0).standard_normal(400000)
    assert rs.quantise(v, helpers, "half_up").var() == pytest.approx(target_white, rel=0.05)


# --- Jumps ---------------------------------------------------------------------------------------

JUMPY = {"theta_f": 0.4, "sigma_f": 0.1, "theta_s": 0.03, "sigma_s": 0.015, "white": 0.0, "lam_per_hour": 1.5,
         "p_drop": 0.7, "mean_drop": 1.0, "mean_rise": 0.6, "jump_min": 1.5}


def test_a_flat_step_is_measured_exactly(cfg):
    x = np.r_[np.zeros(30), np.full(30, 2.5)]
    where, size = rf.detect_jumps(x, cfg)
    assert list(where) == [29] and size[0] == pytest.approx(2.5)


def test_no_jumps_found_in_smooth_noise(cfg):
    x, _ = series({**JUMPY, "lam_per_hour": 0.0}, n_paths=1, n_steps=20000, seed=6)
    assert len(rf.detect_jumps(x.ravel(), cfg)[0]) < 5


@pytest.mark.parametrize("rate", [0.5, 1.0, 1.5])
def test_jump_rate_is_recovered_within_10_percent(cfg, rate):
    x, _ = series({**JUMPY, "lam_per_hour": rate}, n_paths=1, n_steps=200000, seed=7)
    n = x.size
    fit = rf.fit_jumps(x.ravel(), np.ones(n, bool), np.full(n, "sct"), np.full(n, 12), cfg)
    assert fit["lam_per_hour"] == pytest.approx(rate, rel=0.10)
    assert fit["lam_by_sky_hour"]["sct"]["12"] == pytest.approx(rate, rel=0.10)


def test_one_sided_jumps_leave_the_residual_centred():
    rp = {**JUMPY, "p_drop": 0.9, "sigma_f": 0.0, "sigma_s": 0.0, "lam_per_hour": 3.0}
    sim = rs.simulate(rp, 20000, 400, np.random.default_rng(21), init_slow=np.zeros(20000), init_fast=np.zeros(20000))
    assert abs(sim["x_slow"][:, -1].mean()) < 0.03


def test_jump_variance_matches_a_simulation_and_is_removed_from_the_slow_component(cfg):
    rp = {**JUMPY, "sigma_f": 0.0, "sigma_s": 0.0, "lam_per_hour": 2.0}
    sim = rs.simulate(rp, 20000, 300, np.random.default_rng(20), init_slow=np.zeros(20000), init_fast=np.zeros(20000))
    jumps = {"lam_per_hour": 2.0, "active_fraction": 1.0, **{k: JUMPY[k] for k in ("p_drop", "mean_drop", "mean_rise", "jump_min")}}
    assert sim["x_slow"][:, -1].var() == pytest.approx(rf.jump_variance(jumps, JUMPY["theta_s"]), rel=0.05)


def test_jump_sizes_are_recovered(cfg):
    x, _ = series(JUMPY, n_paths=1, n_steps=200000, seed=8)
    where, size = rf.detect_jumps(x.ravel(), cfg)
    drop, rise = -size[size < 0], size[size > 0]
    assert len(drop) / len(size) == pytest.approx(0.7, abs=0.04)
    assert drop.mean() == pytest.approx(1.5 + 1.0, rel=0.1)
    assert rise.mean() == pytest.approx(1.5 + 0.6, rel=0.1)


def test_jumps_are_counted_only_by_day_under_broken_cloud(cfg):
    x = np.r_[np.zeros(30), np.full(30, 3.0), np.full(30, 0.0)]
    n = len(x)
    night = rf.fit_jumps(x, np.zeros(n, bool), np.full(n, "sct"), np.full(n, 2), cfg)
    clear = rf.fit_jumps(x, np.ones(n, bool), np.full(n, "clear"), np.full(n, 12), cfg)
    assert night["lam_per_hour"] == 0.0 and clear["lam_per_hour"] == 0.0


def test_jump_detection_ignores_gaps(cfg):
    x = np.r_[np.zeros(30), np.full(10, np.nan), np.full(30, 4.0)]
    assert len(rf.detect_jumps(x, cfg)[0]) == 0


# --- Tails ---------------------------------------------------------------------------------------

def test_scaled_student_t_innovations_have_unit_variance():
    z = rs.innovations(np.random.default_rng(0), 6.0, 2000000)
    assert z.var() == pytest.approx(1.0, abs=0.02)


def test_gaussian_innovations_get_no_nu(cfg):
    x, _ = series({**TRUE, "theta_s": 0.03, "sigma_s": 0.0, "white": 0.0}, n_paths=1, n_steps=200000, seed=9)
    assert rf.fit_tails(x.ravel(), TRUE["theta_f"], cfg)["nu"] is None


def test_student_t_degrees_of_freedom_are_recovered(cfg):
    rp = {"theta_f": 0.3, "sigma_f": 0.3, "theta_s": 0.03, "sigma_s": 0.0, "white": 0.0, "nu": 5.0}
    x = rs.simulate(rp, 1, 300000, np.random.default_rng(10))["x_fast"].ravel()
    tail = rf.fit_tails(x, 0.3, cfg)
    assert tail["excess_kurtosis"] > 1.0
    assert tail["nu"] == pytest.approx(5.0, rel=0.2)


# --- Sensor values and validation against the floor ------------------------------------------------

def test_simulated_one_minute_values_are_whole_numbers():
    sim = rs.simulate(TRUE, 100, 124, np.random.default_rng(11))
    t = rs.quantise(80.0 + sim["x_slow"] + sim["x_fast"] + sim["white"], helpers, "half_up")
    assert np.array_equal(t, np.round(t))


def test_regime_can_switch_inside_a_path():
    rp = {"theta_f": 0.3, "sigma_f": np.r_[np.zeros(60), np.full(60, 0.5)], "theta_s": 0.03, "sigma_s": 0.0, "white": 0.0}
    sim = rs.simulate(rp, 2000, 120, np.random.default_rng(12), init_slow=np.zeros(2000), init_fast=np.zeros(2000))
    assert (sim["x_fast"][:, 59] == 0).all() and sim["x_fast"][:, -1].var() > 0.2


def test_rolling_mean_matches_a_direct_calculation():
    x = np.arange(10.0)
    r = rs.rolling_mean(x, 5)
    assert np.isnan(r[:4]).all() and r[4] == 2.0 and r[-1] == 7.0


def test_floor_errors_are_zero_for_a_noiseless_path():
    m = 80.0 + np.sin(np.arange(124) / 30.0)
    e_h, e_l = rs.floor_errors(m, m[None, :], 4, 5)
    assert e_h[0] == pytest.approx(0.0) and e_l[0] == pytest.approx(0.0)


def test_fitted_process_reproduces_e_h_and_e_l_of_the_truth(cfg):
    """The Phase 6 check: fit on one sample, simulate, and compare windows from another sample."""
    truth = {**TRUE, "lam_per_hour": 1.0, "p_drop": 0.7, "mean_drop": 0.8, "mean_rise": 0.5, "jump_min": 1.2, "white": 0.1}
    train, seg = series(truth, n_paths=300, n_steps=2000, seed=13)
    n = train.size
    acov = rf.fit_two_exponentials(rf.autocovariance(train.ravel(), 90, seg), cfg)
    acov = {**acov, "white": acov["white"] + cfg["residual"]["quantisation_var"]}
    fitted = rf.fit_regime(train.ravel(), acov, np.ones(n, bool), np.full(n, "sct"), np.full(n, 13), cfg)

    def errors(rp, seed):
        # Burn-in so each window starts with the variance the jumps have built up, as real windows do.
        burn = 400
        m = 80.3 + 1.5 * np.sin(np.arange(124) / 40.0)
        sim = rs.simulate(rp, 20000, burn + 124, np.random.default_rng(seed))
        noise = (sim["x_slow"] + sim["x_fast"] + sim["white"])[:, burn:]
        return rs.floor_errors(m, rs.quantise(m + noise, helpers, "half_up"), 4, 5)

    emp_h, emp_l = errors(truth, 14)
    sim_h, sim_l = errors(fitted, 15)
    assert rf.validate_against_floor(sim_h, emp_h, cfg)["passed"]
    assert rf.validate_against_floor(sim_l, emp_l, cfg)["passed"]
