"""ERA5-only covariates (preregistration §4): domain, soil-moisture anomalies, coupling regime,
warm season, controls and hot-day thresholds. No forecast data are used here."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from landblind.climatology import (
    doy_mean_sd,
    doy_quantile,
    in_season,
    noleap_doy,
    warmest_months,
)
from landblind.config import Config
from landblind.fetch import ERA5_DAILY_VARS, ROOT_ZONE
from landblind.grid import Cells, assign_continents, candidate_land_cells, nearest_utc_offset
from landblind.mirror import Store, open_dataset, require

log = logging.getLogger(__name__)

PERIOD = ("1990-01-01", "2022-12-31")
ANALYSIS_YEARS = (2020, 2022)


def read_cells(store: Store, var: str, time_idx: np.ndarray, cells: Cells,
               block: int = 480) -> np.ndarray:
    """Read ``var`` at ``time_idx`` for the given cells, as float32 (time, cell)."""
    require(store, var, {"time": time_idx})
    da = open_dataset(store)[var]
    if da.dims[1:] != ("longitude", "latitude"):
        raise ValueError(f"unexpected dims {da.dims} for {var}")
    out = np.empty((time_idx.size, cells.size), dtype=np.float32)
    for start in range(0, time_idx.size, block):
        sl = time_idx[start:start + block]
        out[start:start + sl.size] = cells.take(da.isel(time=sl).values)
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return np.interp(qs, cw, values[order])


def build_covariates(cfg: Config) -> xr.Dataset:
    dom = cfg["domain"]
    six, daily = cfg.store("era5_6h"), cfg.store("era5_daily")
    ds6 = open_dataset(six)
    lon, lat = ds6.longitude.values, ds6.latitude.values
    cand = candidate_land_cells(lon, lat, ds6.land_sea_mask.values, dom["land_sea_mask_min"],
                                dom["lat_min"], dom["lat_max"])
    log.info("candidate land cells: %d", cand.size)

    # ---- daily ERA5 ------------------------------------------------------------------------
    times = pd.DatetimeIndex(open_dataset(daily).time.values)
    tsel = np.flatnonzero((times >= PERIOD[0]) & (times <= PERIOD[1]))
    days = times[tsel]
    fields = {}
    for var in ERA5_DAILY_VARS:
        log.info("reading ERA5 daily %s", var)
        fields[var] = read_cells(daily, var, tsel, cand)
    weights = cfg["root_zone_weights"]
    theta = sum(weights[v] * fields[v] for v in ROOT_ZONE) / sum(weights.values())
    theta_l1 = fields[ROOT_ZONE[0]]
    lh_up = -fields["mean_surface_latent_heat_flux"]
    tmax = fields["2m_temperature_max"]
    del fields

    y0, y1 = cfg["climatology_years"]
    clim = (days.year >= y0) & (days.year <= y1)
    doy = noleap_doy(days)
    hw = cfg["doy_half_window"]

    def anomalies(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, sd = doy_mean_sd(x[clim], doy[clim], hw)
        return (x - mean[doy - 1]).astype(np.float32), sd[doy - 1].astype(np.float32)

    theta_anom, theta_sd = anomalies(theta)
    l1_anom, l1_sd = anomalies(theta_l1)
    del theta, theta_l1
    lh_anom, _ = anomalies(lh_up)
    del lh_up
    with np.errstate(invalid="ignore", divide="ignore"):
        s = theta_anom / theta_sd
        s_l1 = l1_anom / l1_sd
    del theta_sd, l1_anom, l1_sd

    # ---- warm season and coupling index (1990-2019 only) -------------------------------------
    months = np.asarray(days.month)
    monthly_tmax = np.stack([np.nanmean(tmax[clim & (months == m)], axis=0) for m in range(1, 13)])
    season_start = warmest_months(monthly_tmax, cfg["warm_season_months"])
    tmax_anom, _ = anomalies(tmax)
    del tmax
    ws = in_season(months, season_start, cfg["warm_season_months"]) & clim[:, None]
    th_ws = np.where(ws, theta_anom, np.nan)
    lh_ws = np.where(ws, lh_anom, np.nan)
    th_c = th_ws - np.nanmean(th_ws, axis=0)
    lh_c = lh_ws - np.nanmean(lh_ws, axis=0)
    sd_ws = np.sqrt(np.nanmean(th_c**2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        tci = np.nanmean(th_c * lh_c, axis=0) / sd_ws

    in_domain = np.isfinite(tci) & (sd_ws >= dom["min_soil_moisture_sd"])
    log.info("domain cells after soil-moisture variability screen: %d of %d",
             int(in_domain.sum()), cand.size)
    cells = cand.subset(in_domain)
    tci, sd_ws, season_start = tci[in_domain], sd_ws[in_domain], season_start[in_domain]
    s, s_l1, tmax_anom = s[:, in_domain], s_l1[:, in_domain], tmax_anom[:, in_domain]

    w = cells.weight
    t1, t2 = weighted_quantile(tci, w, np.array([1 / 3, 2 / 3]))
    regime = np.where(tci >= t2, "water", np.where(tci <= t1, "energy", "middle"))
    dec_edges = weighted_quantile(tci, w, np.linspace(0.1, 0.9, 9))
    decile = np.searchsorted(dec_edges, tci, side="right") + 1

    # ---- 6-hourly ERA5 2 m temperature: control a2 and hot-day thresholds ------------------------
    t6 = pd.DatetimeIndex(ds6.time.values)
    sel6 = np.flatnonzero((t6 >= f"{y0}-01-01") & (t6 <= PERIOD[1] + "T18"))
    log.info("reading ERA5 6-hourly 2 m temperature (%d steps)", sel6.size)
    t2m = read_cells(six, "2m_temperature", sel6, cells)
    t6 = t6[sel6]
    by_hour = {h: (np.flatnonzero(t6.hour == h)) for h in (0, 6, 12, 18)}
    if len({rows.size for rows in by_hour.values()}) != 1:
        raise ValueError("6-hourly ERA5 selection is not complete at all four synoptic hours")
    day_index = pd.DatetimeIndex(t6[by_hour[0]].normalize())
    doy6 = noleap_doy(day_index)
    clim6 = (day_index.year >= y0) & (day_index.year <= y1)

    t00 = t2m[by_hour[0]]
    mean00, _ = doy_mean_sd(t00[clim6], doy6[clim6], hw)
    a2 = (t00 - mean00[doy6 - 1]).astype(np.float32)

    aft_offset, _ = nearest_utc_offset(cells.lon, cfg["afternoon_lst"], 6)
    aft_hour = aft_offset % 24
    t_aft = np.empty((day_index.size, cells.size), dtype=np.float32)
    for h, rows in by_hour.items():
        cols = aft_hour == h
        t_aft[:, cols] = t2m[rows][:, cols]
    q90 = doy_quantile(t_aft[clim6], doy6[clim6], cfg["hot_quantile"], cfg["quantile_half_window"])
    del t2m

    continent = assign_continents(cells, cfg["continents"])
    ds = xr.Dataset(
        {
            "s": (("day", "cell"), s.astype(np.float32)),
            "s_l1": (("day", "cell"), s_l1.astype(np.float32)),
            "tmax_anom": (("day", "cell"), tmax_anom),
            "t2m_00utc_anom": (("day6", "cell"), a2),
            "q90_afternoon": (("doy", "cell"), q90),
            "tci": ("cell", tci.astype(np.float32)),
            "sm_sd_warm_season": ("cell", sd_ws.astype(np.float32)),
            "season_start_month": ("cell", season_start.astype(np.int8)),
            "regime": ("cell", regime.astype(str)),
            "tci_decile": ("cell", decile.astype(np.int8)),
            "continent": ("cell", continent),
            "ilon": ("cell", cells.ilon.astype(np.int16)),
            "ilat": ("cell", cells.ilat.astype(np.int16)),
            "lon": ("cell", cells.lon),
            "lat": ("cell", cells.lat),
            "weight": ("cell", w),
        },
        coords={"day": days.values, "day6": day_index.values, "doy": np.arange(1, 366)},
        attrs={
            "description": "ERA5-only covariates for land-blind (preregistration section 4).",
            "tci_tercile_bounds": [float(t1), float(t2)],
            "n_candidate_cells": cand.size,
        },
    )
    out = cfg.processed / "covariates.nc"
    ds.to_netcdf(out)
    log.info("wrote %s: %d cells (water %d, energy %d)", out, cells.size,
             int((regime == "water").sum()), int((regime == "energy").sum()))
    return ds
