import numpy as np

from kmia2h import target


def test_constant():
    t = np.full(124, 80.0)
    assert target.settle(t, "half_up") == (80, 80)


def test_example_average():
    r = target.reported(np.array([80, 80, 80, 81, 81.0]), "half_up")
    assert np.isclose(target.rolling_mean(np.array([80, 80, 80, 81, 81.0]))[-1], 80.4)
    assert r[-1] == 80


def test_four_valid_tie():
    # four valid values averaging exactly x.5: half_up rounds up
    x = np.array([np.nan, 80, 80, 81, 81.0])
    assert target.rolling_mean(x)[-1] == 80.5
    assert target.reported(x, "half_up")[-1] == 81
    assert target.reported(x, "half_even")[-1] == 80


def test_too_many_nans():
    x = np.array([np.nan, np.nan, 80, 81, 81.0])
    assert np.isnan(target.reported(x, "half_up")[-1])
    t = np.full(124, np.nan)
    with np.errstate(all="ignore"):
        h, l = target.settle(t, "half_up")
    assert np.isnan(h) and np.isnan(l)


def test_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    paths = np.round(80 + np.cumsum(rng.normal(0, 0.3, (50, 124)), axis=1))
    paths[rng.random(paths.shape) < 0.03] = np.nan
    h, l = target.settle(paths, "half_up")
    for i in range(50):
        hi, lo = target.settle(paths[i], "half_up")
        assert (np.isnan(hi) and np.isnan(h[i])) or hi == h[i]
        assert (np.isnan(lo) and np.isnan(l[i])) or lo == l[i]


def test_settle_any_length():
    t = np.r_[np.full(4, 80.0), np.full(30, 82.0)]
    h, l = target.settle(t, "half_up")
    assert (h, l) == (82, 80)


def test_edge():
    assert np.isclose(target.edge(80.5), 0.0)
    assert np.isclose(target.edge(80.0), 0.5)
