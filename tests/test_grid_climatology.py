import numpy as np
import pandas as pd
import pytest

from landblind.climatology import (
    doy_mean_sd,
    doy_quantile,
    in_season,
    noleap_doy,
    warmest_months,
)
from landblind.grid import Cells, assign_continents, local_solar_hour, nearest_utc_offset


@pytest.mark.parametrize(
    ("lon", "step", "offset", "dev"),
    [
        (45.0, 6, 12, 0.0),     # 12 UTC is 15 LST
        (0.0, 6, 12, 3.0),      # tie between 12 and 18 UTC broken towards the earlier step
        (240.0, 6, 24, 1.0),    # 120 W: 00 UTC next day is 16 LST
        (135.0, 12, 12, 6.0),   # 12-hourly model: no step near the afternoon at 135 E
        (225.0, 12, 24, 0.0),   # 135 W: 00 UTC is 15 LST
    ],
)
def test_nearest_utc_offset(lon, step, offset, dev):
    got_offset, got_dev = nearest_utc_offset(np.array([lon]), 15.0, step)
    assert got_offset[0] == offset
    assert got_dev[0] == pytest.approx(dev)


def test_local_solar_hour_wraps():
    assert local_solar_hour(0.0, np.array([-90.0]))[0] == 18.0


def test_noleap_doy_maps_feb29_to_feb28():
    t = pd.DatetimeIndex(["2020-02-28", "2020-02-29", "2020-03-01", "2021-03-01", "2020-12-31"])
    np.testing.assert_array_equal(noleap_doy(t), [59, 59, 60, 60, 365])


def test_doy_climatology_recovers_seasonal_cycle():
    days = pd.date_range("1990-01-01", "2019-12-31", freq="D")
    doy = noleap_doy(days)
    rng = np.random.default_rng(0)
    cycle = 10 * np.sin(2 * np.pi * doy / 365.0)
    x = (cycle + rng.normal(0, 2.0, doy.size))[:, None]
    mean, sd = doy_mean_sd(x, doy, 15)
    expected = 10 * np.sin(2 * np.pi * np.arange(1, 366) / 365.0)
    # A 31-day boxcar damps the annual harmonic by sin(pi*31/365)/(31*sin(pi/365)) ~ 0.988.
    np.testing.assert_allclose(mean[:, 0], 0.988 * expected, atol=0.35)
    assert np.median(sd[:, 0]) == pytest.approx(2.0, rel=0.1)
    q = doy_quantile(x, doy, 0.9, 7)
    assert q.shape == (365, 1)


def test_warmest_months_wraps_year_end():
    monthly = np.array([30, 29, 25, 20, 15, 10, 9, 10, 15, 20, 25, 31], dtype=float)[:, None]
    start = warmest_months(monthly, 3)
    assert start[0] == 12  # December-January-February
    assert in_season(np.array([12, 1, 2, 3]), start, 3)[:, 0].tolist() == [True, True, True, False]


def test_assign_continents_first_match():
    cells = Cells(np.array([0, 1]), np.array([0, 1]), np.array([260.0, 10.0]),
                  np.array([40.0, 50.0]))
    names = assign_continents(cells, {"north_america": [-170, -50, 7, 80],
                                      "europe": [-25, 45, 35, 72]})
    assert names.tolist() == ["north_america", "europe"]
