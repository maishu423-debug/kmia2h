import numpy as np

from kmia2h.data import fetch_goes


class FakeGeo:
    def __init__(self, n=201, dx_km=1.0):
        c = (n - 1) / 2
        yy, xx = np.mgrid[0:n, 0:n]
        east, north = (xx - c) * dx_km, (c - yy) * dx_km
        self.dist_km = np.hypot(east, north)
        az = np.degrees(np.arctan2(east, north)) % 360
        self.sector = (np.floor(az / 30).astype(int)) % 12
        self.ring = np.searchsorted(fetch_goes.RINGS, self.dist_km, side="right") - 1
        self.dx_m = dx_km * 1000
        self.east, self.north = east, north


def test_ring_fractions_exact():
    g = FakeGeo()
    # cloud only in the eastern half: sectors 0-5 (bearings 0-180) fully cloudy, 6-11 clear
    field = (g.east > 0).astype(float)
    field[g.east == 0] = np.where(g.north[g.east == 0] >= 0, 1.0, 0.0)  # bearing 0 (and the centre) is sector 0
    r = fetch_goes.regions(g, field, "cf", "mean")
    for s in range(12):
        for k in range(4):
            assert r[f"cf_s{s:02d}_r{k}"] == (1.0 if s < 6 else 0.0), (s, k)


def test_ring_fraction_half():
    g = FakeGeo()
    field = np.where((g.ring == 2) & (g.sector == 3) & (np.arange(g.ring.size).reshape(g.ring.shape) % 2 == 0), 1.0, 0.0)
    m = (g.ring == 2) & (g.sector == 3)
    expect = field[m].mean()
    assert fetch_goes.regions(g, field, "cf", "mean")["cf_s03_r2"] == expect


def test_motion_recovers_shift():
    rng = np.random.default_rng(0)
    a = (rng.random((101, 101)) > 0.7).astype(float)
    from scipy.ndimage import gaussian_filter
    a = (gaussian_filter(a, 2) > 0.3).astype(float)
    b = np.roll(np.roll(a, 3, axis=1), -2, axis=0)   # 3 px east, 2 px north
    u, v, q = fetch_goes.motion(a, b, 2000.0, 600.0)
    assert abs(u - 3 * 2000 / 600) < 1.5 and abs(v - 2 * 2000 / 600) < 1.5 and q > 0.2
