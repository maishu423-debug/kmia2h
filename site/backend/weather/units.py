"""Unit conversions, humidity and wind components for the ASOS parser and nowcast (no I/O).

Written for this project (the old project's meteo module is not used). Humidity uses the Magnus formula with the
Alduchov-Eskridge (1996) constants. Every function passes None through."""
import math

MAGNUS_A, MAGNUS_B = 17.625, 243.04      # Alduchov & Eskridge (1996), degC


def c_to_f(c):
    return None if c is None else c * 1.8 + 32.0


def ms_to_mph(v):
    return None if v is None else v / 0.44704


def pa_to_hpa(v):
    return None if v is None else v / 100.0


def m_to_miles(v):
    return None if v is None else v / 1609.344


def _gamma(temp_c):
    return MAGNUS_A * temp_c / (MAGNUS_B + temp_c)


def rh_from_t_td(temp_c, dewpoint_c):
    """Relative humidity as a 0-1 fraction from temperature and dew point (degC)."""
    if temp_c is None or dewpoint_c is None:
        return None
    return min(1.0, max(0.0, math.exp(_gamma(dewpoint_c) - _gamma(temp_c))))


def dewpoint_c_from_t_rh(temp_c, rh_frac):
    """Dew point (degC) from temperature (degC) and relative humidity (0-1)."""
    if temp_c is None or rh_frac is None or rh_frac <= 0:
        return None
    g = math.log(rh_frac) + _gamma(temp_c)
    return MAGNUS_B * g / (MAGNUS_A - g)


def wind_to_uv(speed, direction_deg):
    """(speed, direction the wind blows FROM) -> (u east, v north). Calm is (0, 0); unknown direction -> None."""
    if speed is None:
        return None, None
    if speed == 0:
        return 0.0, 0.0
    if direction_deg is None:
        return None, None
    a = math.radians(direction_deg % 360.0)
    return -speed * math.sin(a), -speed * math.cos(a)


def uv_to_speed_dir(u, v):
    """(u, v) -> (speed, direction FROM in degrees); calm has no direction."""
    if u is None or v is None:
        return None, None
    speed = math.hypot(u, v)
    if speed < 1e-9:
        return 0.0, None
    return speed, math.degrees(math.atan2(-u, -v)) % 360.0
