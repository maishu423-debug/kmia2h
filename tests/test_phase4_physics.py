"""Phase 4: every result here has a closed-form or constructed answer."""
import numpy as np
import pandas as pd
import pytest

from kmia2h.physics import cloud_advect, ode, rain, solar, upstream

ORIGIN = pd.Timestamp("2024-07-01 15:00", tz="UTC")
KMIA = {"lat": 25.79, "lon": -80.29, "elevation_m": 3.0}
COEF = {"a": 6e-5, "b": 0.02, "c": 0.0, "kappa": 0.7, "lam": 0.8}


def forcing(n=120, **kw):
    f = {"s_clr": np.zeros(n), "c": np.zeros(n), "t_adv": np.full(n, 80.0), "t_sky": np.full(n, 60.0), "r": np.zeros(n)}
    f.update({k: np.broadcast_to(v, (n,)).astype(float) for k, v in kw.items()})
    return f


def test_zero_forcing_relaxes_exponentially_to_t_adv(cfg):
    p = ode.integrate(90.0, COEF, forcing(), cfg)
    exact = 80.0 + 10.0 * np.exp(-0.02 * np.arange(1, 121))
    assert np.max(np.abs(p["t"] - exact)) < 1e-5


def test_relaxation_reaches_t_adv_at_long_times(cfg):
    assert ode.integrate(70.0, COEF, forcing(n=2000), cfg)["t"].iloc[-1] == pytest.approx(80.0, abs=1e-3)


def test_b_scale_changes_the_rate(cfg):
    p = ode.integrate(90.0, COEF, forcing(b_scale=2.0), cfg)
    assert p["t"].iloc[-1] == pytest.approx(80 + 10 * np.exp(-0.04 * 120), abs=1e-5)


def test_constant_solar_has_known_equilibrium(cfg):
    f = forcing(n=3000, s_clr=800.0, c=0.5)
    expected = 80.0 + 6e-5 * 800.0 * (1 - 0.7 * 0.5) / 0.02
    assert ode.integrate(80.0, COEF, f, cfg)["t"].iloc[-1] == pytest.approx(expected, abs=1e-3)


def test_longwave_uses_kelvin_and_cloud_suppression(cfg):
    coef = {"a": 0.0, "b": 0.0, "c": 3e-4, "kappa": 0.7, "lam": 0.8}
    t, sky, c = 77.0, 41.0, 0.5   # 298.15 K and 278.15 K
    f = {"s_clr": 0.0, "c": c, "t_adv": t, "t_sky": sky, "r": 0.0}
    expected = -3e-4 * cfg["physics"]["sigma_sb"] * (298.15 ** 4 - 278.15 ** 4) * (1 - 0.8 * 0.5)
    assert float(ode.terms(t, coef, f, cfg)["longwave"]) == pytest.approx(expected, rel=1e-9)


def test_euler_and_rk45_agree(cfg):
    f = forcing(s_clr=500.0, c=0.3)
    t = 85.0
    for i in range(120):
        t = ode.euler_step(t, COEF, {k: v[i] for k, v in f.items()}, cfg)
    assert t == pytest.approx(ode.integrate(85.0, COEF, f, cfg)["t"].iloc[-1], abs=0.02)


def test_vectorised_rhs_matches_scalars(cfg):
    coef = {**COEF, "b": np.array([0.01, 0.02, 0.03])}
    f = {"s_clr": 500.0, "c": 0.2, "t_adv": 80.0, "t_sky": 60.0, "r": 0.0}
    got = ode.rhs(np.array([85.0, 86.0, 87.0]), coef, f, cfg)
    for i, (t, b) in enumerate(zip([85.0, 86.0, 87.0], [0.01, 0.02, 0.03])):
        assert got[i] == pytest.approx(float(ode.rhs(t, {**COEF, "b": b}, f, cfg)))


def test_integrate_rejects_nan_forcing(cfg):
    with pytest.raises(ValueError, match="NaN"):
        ode.integrate(80.0, COEF, forcing(t_adv=np.nan), cfg)


def test_sky_temperature_is_below_air_and_rises_with_dewpoint(cfg):
    assert ode.sky_temperature_f(80.0, 60.0, cfg) < ode.sky_temperature_f(80.0, 75.0, cfg) < 80.0


def test_solar_term_is_zero_at_night_and_positive_at_noon(cfg):
    idx = pd.date_range("2024-07-01 00:00", periods=1440, freq="min", tz="UTC")
    s = solar.clear_sky(idx, KMIA)
    assert (s.loc[~s["is_day"], "ghi"] == 0).all()
    assert s["ghi"].max() > 800
    # Solar noon in Miami is near 17:25 UTC.
    assert abs(s["ghi"].idxmax().hour + s["ghi"].idxmax().minute / 60 - 17.4) < 0.3
    tt = ode.terms(80.0, COEF, {"s_clr": s["ghi"].to_numpy(), "c": 0.0, "t_adv": 80.0, "t_sky": 60.0, "r": 0.0}, cfg)
    assert (tt["solar"][~s["is_day"].to_numpy()] == 0).all()


# --- Advection ---------------------------------------------------------------------------------

@pytest.mark.parametrize("u,v,speed,direction", [
    (0, -10, 10, 0), (-10, 0, 10, 90), (0, 10, 10, 180), (10, 0, 10, 270)])
def test_wind_direction_is_where_wind_comes_from(u, v, speed, direction):
    s, d = upstream.wind_speed_dir(u, v)
    assert s == pytest.approx(speed) and d == pytest.approx(direction)


@pytest.mark.parametrize("d,cls", [(30, "onshore"), (150, "onshore"), (151, "other"), (210, "offshore"),
                                   (330, "offshore"), (331, "northerly"), (0, "northerly"), (29.9, "northerly")])
def test_wind_classes(cfg, d, cls):
    assert upstream.wind_class(d, cfg) == cls


def ramp_obs(station, slope=0.1, available_lag=8):
    ts = pd.date_range(ORIGIN - pd.Timedelta(hours=4), ORIGIN + pd.Timedelta(hours=3), freq="5min")
    minutes = (ts - ORIGIN).total_seconds().to_numpy() / 60
    return pd.DataFrame({"station": station, "ts_utc": ts, "tmpf": 80.0 + slope * minutes,
                         "available_at": ts + pd.Timedelta(minutes=available_lag)})


def empty_kmia():
    return pd.DataFrame({"ts_utc": pd.DatetimeIndex([], tz="UTC"), "drct": [], "dwpf": [],
                         "available_at": pd.DatetimeIndex([], tz="UTC")})


def test_advection_lag_is_distance_over_speed(cfg):
    assert upstream.lag_minutes(30.0, 10.0, cfg) == pytest.approx(50.0)
    assert upstream.lag_minutes(300.0, 1.0, cfg) == pytest.approx(120.0)
    assert upstream.lag_minutes(30.0, 0.0, cfg) == pytest.approx(120.0)


def test_t_adv_is_the_lagged_upstream_series_for_a_synthetic_wind(cfg):
    # Station due north at 30 km, northerly wind 10 m/s: the lag is 50 minutes.
    d_lat = 30.0 / upstream.EARTH_RADIUS_KM * 180.0 / np.pi
    meta = {"KOPF": {"lat": KMIA["lat"] + d_lat, "lon": KMIA["lon"]}}
    out = upstream.build(ORIGIN, 40, ramp_obs("KOPF"), empty_kmia(), meta, KMIA, 0.0, -10.0, cfg, t_last=75.0)
    assert out["wind_class"] == "northerly"
    assert out["t_adv"] == pytest.approx(80.0 + 0.1 * (np.arange(1, 41) - 50.0), abs=1e-6)


def test_no_look_ahead_in_t_adv(cfg):
    obs = ramp_obs("KOPF")
    late = obs.copy()
    late.loc[late["available_at"] > ORIGIN, "tmpf"] = 500.0
    a = upstream.lagged_series(obs, ORIGIN, "KOPF", 10.0, 60, cfg)
    b = upstream.lagged_series(late, ORIGIN, "KOPF", 10.0, 60, cfg)
    assert np.array_equal(a, b)
    # Newest usable report is valid at origin - 10 (origin - 5 only becomes available at origin + 3).
    assert a[-1] == pytest.approx(80.0 + 0.1 * -10.0)


def test_t_adv_falls_back_to_hrrr_then_last_temperature(cfg):
    obs = ramp_obs("NOPE")
    meta = {"KOPF": {"lat": 25.9, "lon": -80.28}}
    hrrr = np.full(10, 83.0)
    assert (upstream.build(ORIGIN, 10, obs, empty_kmia(), meta, KMIA, 0, -5, cfg, 75.0, hrrr_t2m=hrrr)["t_adv"] == 83.0).all()
    assert (upstream.build(ORIGIN, 10, obs, empty_kmia(), meta, KMIA, 0, -5, cfg, 75.0)["t_adv"] == 75.0).all()


def test_missing_station_drops_out_of_the_weights(cfg):
    meta = {"KOPF": {"lat": 25.9, "lon": -80.28}, "KHWO": {"lat": 26.0, "lon": -80.24}}
    out = upstream.build(ORIGIN, 10, ramp_obs("KOPF", slope=0.0), empty_kmia(), meta, KMIA, 0, -5, cfg, 75.0)
    assert out["t_adv"] == pytest.approx(80.0)


def test_inverse_distance_weights(cfg):
    a, b = ramp_obs("KOPF", slope=0.0), ramp_obs("KHWO", slope=0.0)
    b["tmpf"] = 90.0
    d = 1.0 / upstream.EARTH_RADIUS_KM * 180.0 / np.pi
    meta = {"KOPF": {"lat": KMIA["lat"] + 10 * d, "lon": KMIA["lon"]}, "KHWO": {"lat": KMIA["lat"] + 30 * d, "lon": KMIA["lon"]}}
    out = upstream.build(ORIGIN, 5, pd.concat([a, b]), empty_kmia(), meta, KMIA, 0, -10, cfg, 75.0)
    assert out["t_adv"] == pytest.approx((80 / 10 + 90 / 30) / (1 / 10 + 1 / 30), abs=1e-3)


def test_onshore_prefers_buoys_and_falls_back_to_kfll(cfg):
    meta = {"VAKF1": {"lat": 25.73, "lon": -80.16}, "KFLL": {"lat": 26.07, "lon": -80.15}}
    kfll = ramp_obs("KFLL", slope=0.0)
    out = upstream.build(ORIGIN, 5, kfll, empty_kmia(), meta, KMIA, -5, 0, cfg, 75.0)
    assert out["wind_class"] == "onshore" and out["t_adv"] == pytest.approx(80.0)
    buoy = ramp_obs("VAKF1", slope=0.0)
    buoy["tmpf"] = 84.0
    assert upstream.build(ORIGIN, 5, pd.concat([kfll, buoy]), empty_kmia(), meta, KMIA, -5, 0, cfg, 75.0)["t_adv"] == pytest.approx(84.0)


def sea_breeze_obs(dews, dirs):
    ts = pd.date_range(ORIGIN - pd.Timedelta(minutes=25), ORIGIN - pd.Timedelta(minutes=10), periods=len(dews))
    return pd.DataFrame({"ts_utc": ts, "drct": dirs, "dwpf": dews, "available_at": ts + pd.Timedelta(minutes=8)})


def test_observed_sea_breeze_needs_turn_and_dew_rise(cfg):
    assert upstream.sea_breeze_onset(ORIGIN, 10, sea_breeze_obs([70, 72.5], [250, 90]), 250.0, None, cfg) == 0
    assert upstream.sea_breeze_onset(ORIGIN, 10, sea_breeze_obs([70, 71], [250, 90]), 250.0, None, cfg) is None
    assert upstream.sea_breeze_onset(ORIGIN, 10, sea_breeze_obs([70, 73], [90, 90]), 90.0, None, cfg) is None


def test_forecast_sea_breeze_switches_t_adv_and_b(cfg):
    hrrr_dir = np.r_[np.full(30, 250.0), np.full(30, 90.0)]
    d = 20.0 / upstream.EARTH_RADIUS_KM * 180.0 / np.pi
    meta = {"VAKF1": {"lat": KMIA["lat"], "lon": KMIA["lon"] + d}, "KTMB": {"lat": KMIA["lat"], "lon": KMIA["lon"] - d}}
    marine, land = ramp_obs("VAKF1", slope=0.0), ramp_obs("KTMB", slope=0.0)
    marine["tmpf"], land["tmpf"] = 79.0, 90.0
    out = upstream.build(ORIGIN, 60, pd.concat([marine, land]), empty_kmia(), meta, KMIA, 8, 2, cfg, 75.0, hrrr_dir=hrrr_dir)
    assert out["onset"] == 30
    assert out["t_adv"][:30] == pytest.approx(90.0)
    assert out["t_adv"][30:] == pytest.approx(79.0)
    p = cfg["physics"]["prior"]
    assert (out["b_scale"][:30] == 1).all()
    assert out["b_scale"][30:] == pytest.approx(p["b_onshore"] / p["b"])


# --- Cloud -------------------------------------------------------------------------------------

def field(cfg, value=0.0):
    c = cfg["physics"]["cloud"]
    return np.full((c["n_sectors"], c["n_rings"]), value, dtype=float)


def test_cloud_arrives_when_the_ring_distance_is_covered(cfg):
    f = field(cfg)
    f[0, 2] = 1.0   # the 20-40 km ring, due north
    # Cloud moving south at 10 m/s covers 0.6 km/min: the 30 km ring centre arrives at 50 min.
    c = cloud_advect.project(0.0, f, 0.0, -10.0, 120, cfg)[0]
    assert c[49] == pytest.approx(1.0)
    assert c[0] == pytest.approx(0.0)
    assert c[24] == pytest.approx(0.0)  # 15 km centre has no cloud
    assert c[36] == pytest.approx((22.2 - 15) / 15, abs=1e-9)


def test_cloud_uses_the_upwind_sector_not_the_downwind_one(cfg):
    f = field(cfg)
    f[6, 1] = 1.0   # sector 180-210 deg is the south
    assert cloud_advect.project(0.0, f, 0.0, -10.0, 60, cfg)[0].max() == 0.0
    assert cloud_advect.project(0.0, f, 0.0, 10.0, 60, cfg)[0].max() == pytest.approx(1.0)


def test_calm_air_keeps_the_station_value(cfg):
    assert cloud_advect.project(0.4, field(cfg, 1.0), 0.0, 0.0, 30, cfg)[0] == pytest.approx(0.4)


def test_cloud_relaxes_to_hrrr_beyond_the_outer_ring(cfg):
    n = 120
    c = cloud_advect.project(1.0, field(cfg, 1.0), 0.0, -30.0, n, cfg, tcdc=np.zeros(n))[0]
    # 30 m/s covers 80 km in 44.4 min.
    assert c[43] == pytest.approx(1.0)
    assert c[73] == pytest.approx(np.exp(-(74 - 80 / 1.8) / 30.0), abs=1e-9)


def test_tcdc_in_percent_is_rejected(cfg):
    with pytest.raises(ValueError, match="fraction"):
        cloud_advect.project(0.0, field(cfg), 0.0, -5.0, 10, cfg, tcdc=np.full(10, 80.0))


def test_project_is_vectorised_over_motion(cfg):
    f = field(cfg)
    f[0, 2] = 1.0
    both = cloud_advect.project(0.0, f, np.array([0.0, 0.0]), np.array([-10.0, -20.0]), 120, cfg)
    assert both[0, 49] == pytest.approx(1.0) and both[1, 24] == pytest.approx(1.0)


def test_motion_vector_quality_switch(cfg):
    good = {"motion_u": 3.0, "motion_v": 4.0, "motion_q": 0.9}
    assert cloud_advect.motion_vector(good, 1.0, 2.0, cfg) == (3.0, 4.0, "goes")
    assert cloud_advect.motion_vector({**good, "motion_q": 0.1}, 1.0, 2.0, cfg) == (1.5, 3.0, "hrrr_wind")
    assert cloud_advect.motion_vector(None, 1.0, 2.0, cfg)[2] == "hrrr_wind"


def test_convective_growth_only_in_wet_season_midday(cfg):
    day = pd.date_range("2024-07-01 16:00", periods=3, freq="h", tz="UTC")   # 11:00 LST onward
    assert cloud_advect.convective_growth(day, np.full(3, 0.5), 0.2, cfg) == pytest.approx(0.3)
    night = pd.date_range("2024-07-01 02:00", periods=3, freq="h", tz="UTC")
    assert (cloud_advect.convective_growth(night, np.full(3, 0.5), 0.2, cfg) == 0).all()
    dry = pd.date_range("2024-01-15 16:00", periods=3, freq="h", tz="UTC")
    assert (cloud_advect.convective_growth(dry, np.full(3, 0.5), 0.2, cfg) == 0).all()


def test_kappa_class(cfg):
    c = cfg["physics"]["cloud"]
    assert cloud_advect.kappa_value(1.0, 2000.0, True, cfg) == c["kappa_thin"]
    assert cloud_advect.kappa_value(10.0, 9000.0, True, cfg) == c["kappa_thin"]
    assert cloud_advect.kappa_value(10.0, 2000.0, True, cfg) == c["kappa_thick"]
    assert cloud_advect.kappa_value(1.0, 2000.0, False, cfg) == c["kappa_thick"]   # night ignores optical depth


@pytest.mark.parametrize("code,value", [("CLR", 0.0), ("FEW", 0.2), ("SCT", 0.4), ("BKN", 0.75), ("OVC", 1.0), ("VV", 1.0)])
def test_metar_fallback_when_goes_is_missing(cfg, code, value):
    valid = pd.date_range(ORIGIN, periods=5, freq="min")
    out = cloud_advect.build(valid, None, 1.0, 2.0, True, code, cfg)
    p, c = cfg["physics"]["prior"], cfg["physics"]["cloud"]
    assert out["source"] == "metar" and out["c"] == pytest.approx(value)
    assert out["kappa_scale"] == pytest.approx(c["kappa_thick"] / p["kappa"])


# --- Rain --------------------------------------------------------------------------------------

def test_rain_pulse_produces_the_requested_drop_and_recovery(cfg):
    n = 120
    b = cfg["physics"]["prior"]["b"]
    cells = pd.DataFrame({"arrival_min": [20.0], "drop_f": [4.0]})
    coef = {"a": 0.0, "b": b, "c": 0.0, "kappa": 0.0, "lam": 0.0}
    r = rain.rate(cells, n, b, cfg)
    p = ode.integrate(80.0, coef, forcing(n, r=r), cfg)
    g, _ = rain.shape(np.arange(1, n + 1), 20.0, cfg)
    assert np.max(np.abs((80.0 - p["t"].to_numpy()) - 4.0 * g)) < 0.25
    assert p["t"].min() == pytest.approx(76.0, abs=0.25)


def test_no_cells_means_no_cooling(cfg):
    assert (rain.rate(None, 30, 0.01, cfg) == 0).all()
    assert (rain.rate_paths(None, 30, 5, 0.01, np.random.default_rng(0), cfg) == 0).all()


def test_cell_arrival_time_is_geometric(cfg):
    # Sector 0 centre is at bearing 15 deg, ring 1 centre at 15 km, cell moving straight at KMIA.
    b = np.radians(15.0)
    row = {"maxdbz_s00_r1": 45.0, "frac35_s00_r1": 0.2, "motion_u": -10 * np.sin(b), "motion_v": -10 * np.cos(b)}
    cells = rain.find_cells(row, 120, cfg)
    assert len(cells) == 1
    assert cells["arrival_min"].iloc[0] == pytest.approx(np.ceil((15 - 5) / 0.6))
    assert cells["drop_f"].iloc[0] == pytest.approx(0.1 * 10 + 5.0 * 0.2)


def test_cell_moving_away_does_not_arrive(cfg):
    b = np.radians(15.0)
    row = {"maxdbz_s00_r1": 45.0, "frac35_s00_r1": 0.2, "motion_u": 10 * np.sin(b), "motion_v": 10 * np.cos(b)}
    assert rain.find_cells(row, 120, cfg).empty


def test_weak_cell_and_far_cell_are_ignored(cfg):
    b = np.radians(15.0)
    motion = {"motion_u": -10 * np.sin(b), "motion_v": -10 * np.cos(b)}
    assert rain.find_cells({"maxdbz_s00_r1": 30.0, **motion}, 120, cfg).empty
    assert rain.find_cells({"maxdbz_s00_r3": 50.0, **motion}, 120, cfg).empty  # 60 km > 40 km search


def test_rain_path_perturbation_has_the_configured_spread(cfg):
    cells = pd.DataFrame({"arrival_min": [40.0], "drop_f": [4.0]})
    r = rain.rate_paths(cells, 120, 4000, 0.01, np.random.default_rng(1), cfg)
    onset = (r > 0).argmax(axis=1) + 1
    assert onset.std() == pytest.approx(cfg["physics"]["rain"]["arrival_sigma_min"], rel=0.1)
