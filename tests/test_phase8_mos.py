"""Phase 8: MOS, quantile-to-bin conversion, conformal calibration and blending on known cases."""
import copy

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from kmia2h.mos import conformal, predict, train

TAUS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]


def normal_q(mu, sigma, taus=TAUS):
    return stats.norm(mu, sigma).ppf(taus)


# --- Conformal -----------------------------------------------------------------------------------

def biased(n, bias, sigma=1.0, seed=0):
    """Forecast quantiles of the right shape, but shifted by `bias` from the truth."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(70, 90, n)
    y = rng.normal(mu, sigma)
    return np.column_stack([mu + bias + sigma * stats.norm.ppf(t) for t in TAUS]), y


def test_conformal_removes_a_known_bias(cfg):
    q_cal, y_cal = biased(5000, bias=2.0, seed=1)
    q_test, y_test = biased(20000, bias=2.0, seed=2)
    groups = ["wet|True"] * len(y_cal)
    params = conformal.fit(q_cal, y_cal, TAUS, groups, cfg)
    fixed = conformal.apply(q_test, ["wet|True"] * len(y_test), params, cfg)
    median = fixed[:, TAUS.index(0.5)]
    assert (median - y_test).mean() == pytest.approx(0.0, abs=0.06)
    assert (q_test[:, TAUS.index(0.5)] - y_test).mean() == pytest.approx(2.0, abs=0.05)   # it was biased


@pytest.mark.parametrize("lower", [0.02, 0.05, 0.10, 0.20])
def test_conformal_restores_nominal_coverage(cfg, lower):
    q_cal, y_cal = biased(5000, bias=1.5, seed=3)
    q_test, y_test = biased(20000, bias=1.5, seed=4)
    params = conformal.fit(q_cal, y_cal, TAUS, ["g"] * len(y_cal), cfg)
    assert conformal.coverage(q_test, y_test, TAUS, lower) < 1 - 2 * lower - 0.03
    fixed = conformal.apply(q_test, ["g"] * len(y_test), params, cfg)
    assert conformal.coverage(fixed, y_test, TAUS, lower) == pytest.approx(1 - 2 * lower, abs=0.025)


def test_conformal_fixes_a_forecast_that_is_too_narrow(cfg):
    rng = np.random.default_rng(5)
    def sample(n):
        mu = rng.uniform(70, 90, n)
        return np.column_stack([mu + 0.4 * stats.norm.ppf(t) for t in TAUS]), rng.normal(mu, 1.0, n)
    q_cal, y_cal = sample(5000)
    q_test, y_test = sample(20000)
    params = conformal.fit(q_cal, y_cal, TAUS, ["g"] * 5000, cfg)
    fixed = conformal.apply(q_test, ["g"] * 20000, params, cfg)
    assert conformal.coverage(fixed, y_test, TAUS, 0.05) == pytest.approx(0.90, abs=0.025)


def test_each_group_gets_its_own_correction(cfg):
    qa, ya = biased(3000, bias=2.0, seed=6)
    qb, yb = biased(3000, bias=-1.0, seed=7)
    q = np.vstack([qa, qb])
    y = np.r_[ya, yb]
    groups = conformal.group_key(["wet"] * 3000 + ["dry"] * 3000, [True] * 6000)
    params = conformal.fit(q, y, TAUS, groups, cfg)
    median = TAUS.index(0.5)
    assert params["offsets"]["wet|True"][median] == pytest.approx(-2.0, abs=0.1)
    assert params["offsets"]["dry|True"][median] == pytest.approx(1.0, abs=0.1)


def test_small_or_unseen_group_uses_the_pooled_correction(cfg):
    q, y = biased(3000, bias=2.0, seed=8)
    groups = np.array(["big"] * 2990 + ["tiny"] * 10, dtype=object)
    params = conformal.fit(q, y, TAUS, groups, cfg)
    assert "tiny" not in params["offsets"]
    a = conformal.apply(q[:1], ["tiny"], params, cfg)
    b = conformal.apply(q[:1], ["never_seen"], params, cfg)
    assert np.array_equal(a, b)


def test_conformal_output_never_crosses(cfg):
    q, y = biased(300, bias=0.0, seed=9)
    crossed = q.copy()
    crossed[:, 3], crossed[:, 4] = q[:, 4], q[:, 3]
    params = conformal.fit(crossed, y, TAUS, ["g"] * 300, cfg)
    assert (np.diff(conformal.apply(crossed, ["g"] * 300, params, cfg), axis=1) >= 0).all()


def test_conformal_params_are_plain_data(cfg):
    import json
    q, y = biased(200, bias=1.0, seed=10)
    json.dumps(conformal.fit(q, y, TAUS, ["g"] * 200, cfg))


# --- Quantiles to bins ----------------------------------------------------------------------------

def test_cdf_passes_through_the_quantiles_and_is_monotone(cfg):
    q = normal_q(80, 1.5)
    assert predict.cdf(q, TAUS, q, cfg) == pytest.approx(np.array(TAUS), abs=1e-5)
    x = np.linspace(60, 100, 4001)
    f = predict.cdf(x, TAUS, q, cfg)
    assert (np.diff(f) >= -1e-12).all() and f[0] < 1e-3 and f[-1] > 1 - 1e-3


def test_bin_probabilities_match_a_known_normal(cfg):
    mu, sigma = 80.2, 1.2
    ks, p = predict.bin_table(TAUS, normal_q(mu, sigma), cfg)
    d = stats.norm(mu, sigma)
    exact = d.cdf(ks + 0.5) - d.cdf(ks - 0.5)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(p - exact)) < 0.012


def test_exponential_tail_goes_through_the_outer_two_quantiles(cfg):
    q = normal_q(80, 1.5)
    beta = (q[1] - q[0]) / np.log(TAUS[1] / TAUS[0])
    x = q[0] - 2.0
    assert predict.cdf(x, TAUS, q, cfg) == pytest.approx(TAUS[0] * np.exp(-2.0 / beta), rel=1e-4)


def test_repeated_quantiles_still_give_a_valid_table(cfg):
    q = np.r_[np.full(6, 80.0), np.full(7, 80.0)]
    ks, p = predict.bin_table(TAUS, q, cfg)
    assert p.sum() == pytest.approx(1.0) and ks[p.argmax()] == 80 and p.min() >= 0


# --- Copula ---------------------------------------------------------------------------------------

def marginals(cfg, mu_h=82.0, mu_l=74.0):
    ks_h, p_h = predict.bin_table(TAUS, normal_q(mu_h, 1.0), cfg)
    ks_l, p_l = predict.bin_table(TAUS, normal_q(mu_l, 1.0), cfg)
    return ks_h, p_h, ks_l, p_l


def test_zero_correlation_gives_the_product_of_marginals(cfg):
    ks_h, p_h, ks_l, p_l = marginals(cfg)
    table = predict.joint_table(ks_h, p_h, ks_l, p_l, 0.0, cfg)
    assert np.max(np.abs(table - np.outer(p_h, p_l))) < 1e-4


def test_joint_table_preserves_marginals_and_sums_to_one(cfg):
    ks_h, p_h, ks_l, p_l = marginals(cfg)
    table = predict.joint_table(ks_h, p_h, ks_l, p_l, 0.6, cfg)
    assert table.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(table.sum(axis=1) - p_h)) < 2e-3
    assert np.max(np.abs(table.sum(axis=0) - p_l)) < 2e-3


def test_joint_table_matches_the_bivariate_normal(cfg):
    """Gaussian marginals far apart, so no cell is impossible: the table is a bivariate normal."""
    rho = 0.7
    ks_h, ks_l = np.arange(80, 101), np.arange(60, 81)
    def marginal(ks, mu):
        d = stats.norm(mu, 2.0)
        p = d.cdf(ks + 0.5) - d.cdf(ks - 0.5)
        return p / p.sum()
    table = predict.joint_table(ks_h, marginal(ks_h, 90.0), ks_l, marginal(ks_l, 70.0), rho, cfg)
    z0, z1 = -0.25, 0.25    # the central bin, in standard units
    bvn = stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]], seed=0)
    exact = bvn.cdf([z1, z1]) - bvn.cdf([z0, z1]) - bvn.cdf([z1, z0]) + bvn.cdf([z0, z0])
    assert table[ks_h == 90, ks_l == 70][0] == pytest.approx(exact, rel=0.02)


def test_marginals_survive_when_high_and_low_overlap(cfg):
    """Removing the impossible cells must not move the marginals the forecast reports."""
    ks_h, p_h = predict.bin_table(TAUS, normal_q(80.5, 0.8), cfg)
    ks_l, p_l = predict.bin_table(TAUS, normal_q(79.5, 0.8), cfg)
    table = predict.joint_table(ks_h, p_h, ks_l, p_l, 0.5, cfg)
    assert table[ks_h[:, None] < ks_l[None, :]].sum() == 0.0
    assert np.max(np.abs(table.sum(axis=1) - p_h)) < 0.01
    assert np.max(np.abs(table.sum(axis=0) - p_l)) < 0.01


def test_high_below_low_is_impossible(cfg):
    ks = np.arange(75, 86)
    p = np.ones(len(ks)) / len(ks)
    table = predict.joint_table(ks, p, ks, p, 0.3, cfg)
    assert table[ks[:, None] < ks[None, :]].sum() == 0.0 and table.sum() == pytest.approx(1.0)


def test_copula_rho_recovers_a_known_correlation():
    rng = np.random.default_rng(1)
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 100000)
    assert predict.copula_rho(z[:, 0] ** 3, np.exp(z[:, 1]), 0.999) == pytest.approx(0.6, abs=0.01)
    assert predict.copula_rho(np.ones(10), np.arange(10.0), 0.999) == 0.0


def test_joint_table_is_deterministic(cfg):
    ks_h, p_h, ks_l, p_l = marginals(cfg)
    assert np.array_equal(predict.joint_table(ks_h, p_h, ks_l, p_l, 0.4, cfg), predict.joint_table(ks_h, p_h, ks_l, p_l, 0.4, cfg))


def test_dict_conversions(cfg):
    ks = np.array([80, 81])
    d = predict.to_dict(ks, np.array([0.25, 0.75]))
    assert d == {"80": 0.25, "81": 0.75}
    assert list(predict.on_bins(d, np.array([79, 80]))) == [0.0, 0.25]
    assert predict.joint_to_dict(ks, ks, np.array([[0.5, 0.0], [0.25, 0.25]])) == {"80|80": 0.5, "81|80": 0.25, "81|81": 0.25}


def test_blended_marginal_keeps_physics_mass_outside_the_grid():
    ks = np.array([80, 81])
    p = predict.blended_marginal({"79": 0.2, "80": 0.3, "81": 0.3, "82": 0.2}, ks, np.array([0.5, 0.5]), 0.0)
    assert p.sum() == pytest.approx(1.0) and p[0] == pytest.approx(0.5)


# --- Blending -------------------------------------------------------------------------------------

def tables(n, seed):
    """Truth comes from bin probabilities `good`; the other forecast is badly biased."""
    rng = np.random.default_rng(seed)
    good = np.tile(np.array([0.1, 0.8, 0.1]), (n, 1))
    bad = np.tile(np.array([0.8, 0.1, 0.1]), (n, 1))
    return good, bad, rng.choice(3, size=n, p=[0.1, 0.8, 0.1])


def test_blend_weight_goes_to_the_correct_side(cfg):
    good, bad, y = tables(4000, 1)
    groups = ["g"] * 4000
    assert conformal.fit_blend(good, bad, y, groups, cfg)["g"] == 1.0
    assert conformal.fit_blend(bad, good, y, groups, cfg)["g"] == 0.0


def test_blend_finds_an_interior_weight(cfg):
    rng = np.random.default_rng(2)
    a, b = np.array([0.6, 0.3, 0.1]), np.array([0.1, 0.3, 0.6])
    truth = 0.5 * a + 0.5 * b
    y = rng.choice(3, size=40000, p=truth)
    w = conformal.fit_blend(np.tile(a, (40000, 1)), np.tile(b, (40000, 1)), y, ["g"] * 40000, cfg)["g"]
    assert w == pytest.approx(0.5, abs=0.05)


def test_blended_log_loss_is_no_worse_than_either_input(cfg):
    good, bad, y = tables(4000, 3)
    w = conformal.fit_blend(good, bad, y, ["g"] * 4000, cfg)["g"]
    blended = w * good + (1 - w) * bad
    assert conformal.log_loss(blended, y, cfg) <= conformal.log_loss(good, y, cfg) + 1e-12
    assert conformal.log_loss(blended, y, cfg) <= conformal.log_loss(bad, y, cfg) + 1e-12


def test_extreme_weights_are_flagged():
    assert conformal.blend_flags({"a": 1.0, "b": 0.0, "c": 0.5}) == {"a": "mos_only", "b": "physics_only"}


def test_blend_group_fallback(cfg):
    assert conformal.blend_weight({"__all__": 0.3, "g": 0.7}, "g", cfg) == 0.7
    assert conformal.blend_weight({"__all__": 0.3, "g": 0.7}, "unknown", cfg) == 0.3


def test_reliability_of_a_calibrated_forecast(cfg):
    rng = np.random.default_rng(4)
    a = rng.dirichlet([2, 2, 2], 30000)
    y = np.array([rng.choice(3, p=row) for row in a])
    for row in conformal.reliability(a, y):
        assert abs(row["observed"] - row["predicted"]) < 0.05, row


# --- LightGBM ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mos_cfg():
    import helpers
    c = copy.deepcopy(helpers.load_cfg())
    c["mos"]["max_rounds"] = 150
    c["mos"]["early_stopping_rounds"] = 30
    c["mos"]["params"].update({"num_leaves": 15, "learning_rate": 0.1})
    c["mos"]["quantiles"] = [0.05, 0.25, 0.5, 0.75, 0.95]
    c["mos"]["cv_folds"], c["mos"]["embargo_min"] = 3, 120
    c["mos"]["regularise"] = {**c["mos"].get("regularise", {}), "enabled": False}
    c["mos"]["params_by_target"] = {}
    return c


def mos_frame(n, seed):
    """Physics quantiles carry a 1.5 degF warm bias at night and a too-narrow spread."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(70, 90, n)
    night = rng.random(n) < 0.5
    d = {"origin": pd.date_range("2022-01-01", periods=n, freq="30min", tz="UTC")}
    for lvl in (5, 25, 50, 75, 95):
        z = stats.norm.ppf(lvl / 100)
        d[f"q_max_{lvl:02d}"] = mu + 1.5 * night + 0.5 * z
        d[f"q_min_{lvl:02d}"] = mu - 3 + 0.5 * z
    frame = pd.DataFrame(d)
    frame["regime_night"] = night.astype(float)
    frame["avail_goes"] = 1.0
    frame["h_r"] = rng.normal(mu, 1.0)
    frame["l_r"] = rng.normal(mu - 3.0, 1.0)
    frame["t_last"] = mu + rng.normal(0, 3, n)
    return frame


@pytest.fixture(scope="module")
def fitted(mos_cfg):
    return train.train(mos_frame(5000, 1), "h_r", mos_cfg)


def test_time_folds_respect_the_embargo():
    t = pd.Series(pd.date_range("2022-01-01", periods=600, freq="30min", tz="UTC"))
    folds = train.time_folds(t, 3, 120)
    assert len(folds) == 3
    for tr, va in folds:
        assert not set(tr) & set(va)
        lo, hi = t.iloc[va].min(), t.iloc[va].max()
        gaps = t.iloc[tr]
        assert not ((gaps >= lo - pd.Timedelta(minutes=120)) & (gaps <= hi + pd.Timedelta(minutes=120))).any()
    assert sorted(np.concatenate([va for _, va in folds])) == list(range(600))


def test_missing_physics_quantiles_are_rejected(mos_cfg):
    with pytest.raises(ValueError, match="physics"):
        train.feature_list(pd.DataFrame({"origin": [1]}), "h_r", mos_cfg)


def test_mos_beats_raw_physics_on_held_out_data(fitted):
    test = mos_frame(3000, 2)
    q = predict.predict_quantiles(fitted, test)
    i = fitted.taus.index(0.5)
    raw = train.pinball(test["h_r"], test["q_max_50"], 0.5)
    assert train.pinball(test["h_r"], q[:, i], 0.5) < raw - 0.1


def test_mos_learns_the_regime_bias(fitted):
    test = mos_frame(3000, 3)
    q = predict.predict_quantiles(fitted, test)[:, fitted.taus.index(0.5)]
    night = test["regime_night"] == 1.0
    assert (q[night] - test.loc[night, "h_r"]).mean() == pytest.approx(0.0, abs=0.25)


def test_predicted_quantiles_do_not_cross(fitted):
    q = predict.predict_quantiles(fitted, mos_frame(1000, 4))
    assert (np.diff(q, axis=1) >= 0).all()


def test_predictions_rise_with_the_physics_quantiles(fitted, mos_cfg):
    base = mos_frame(500, 5)
    up = base.copy()
    for c in train.physics_columns("h_r", mos_cfg):
        up[c] += 2.0
    i = fitted.taus.index(0.5)
    before, after = predict.predict_quantiles(fitted, base)[:, i], predict.predict_quantiles(fitted, up)[:, i]
    assert (after >= before - 1e-9).all()
    assert (after - before).mean() > 1.0
    # The trees alone respect the constraint, apart from the physics-median anchor.
    raw_before = fitted.boosters[0.5].predict(train.matrix(base, fitted.features), raw_score=True)
    raw_after = fitted.boosters[0.5].predict(train.matrix(up, fitted.features), raw_score=True)
    assert (raw_after >= raw_before - 1e-9).all()


def test_optional_columns_can_be_absent(fitted):
    test = mos_frame(50, 6).drop(columns=["t_last", "avail_goes"])
    assert np.isfinite(predict.predict_quantiles(fitted, test)).all()


def test_end_to_end_post_processing_tables_sum_to_one(cfg, fitted):
    test = mos_frame(200, 7)
    q = predict.predict_quantiles(fitted, test)
    cal = conformal.fit(q, test["h_r"].to_numpy(), fitted.taus, ["g"] * 200, cfg)
    fixed = conformal.apply(q, ["g"] * 200, cal, cfg)
    ks, p = predict.bin_table(fitted.taus, fixed[0], cfg)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    ks_l, p_l = predict.bin_table(fitted.taus, fixed[0] - 6.0, cfg)
    joint = predict.joint_table(ks, p, ks_l, p_l, 0.5, cfg)
    assert joint.sum() == pytest.approx(1.0, abs=1e-9)


def test_regularisation_selection_prefers_the_simpler_model_on_noise(mos_cfg):
    """Pure-noise extra features: the heavily regularised candidate must win out of fold, and the
    chosen params are carried into the fitted model."""
    c = copy.deepcopy(mos_cfg)
    c["mos"]["quantiles"] = [0.5]
    c["mos"]["regularise"] = {"enabled": True, "select_taus": [0.5],
                              "grid": [{"num_leaves": 63, "min_data_in_leaf": 2, "learning_rate": 0.3},
                                       {"num_leaves": 4, "min_data_in_leaf": 200, "lambda_l2": 10.0}]}
    f = mos_frame(1500, 7)
    rng = np.random.default_rng(0)
    for j in range(20):
        f[f"regime_noise{j}"] = rng.normal(size=len(f))
    chosen, table = train.select_params(f, "h_r", c)
    assert len(table) == 2 and chosen["num_leaves"] == 4
    assert table[1]["cv_pinball"] < table[0]["cv_pinball"]
    m = train.train(f, "h_r", c)
    assert m.params["num_leaves"] == 4 and m.selection is not None
