"""Forecast and truth panels (preregistration §4.6–4.9): for every model pair, arrays of shape
(initialisation, lead_day, cell) at each cell's afternoon and night target times."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from landblind.config import Config
from landblind.grid import Cells, nearest_utc_offset
from landblind.mirror import Store, open_dataset, require

log = logging.getLogger(__name__)

G = 9.80665  # m s-2, converts geopotential to geopotential height


def cells_from_covariates(cov: xr.Dataset) -> Cells:
    return Cells(cov.ilon.values.astype(int), cov.ilat.values.astype(int), cov.lon.values,
                 cov.lat.values)


def target_offsets(cells: Cells, target_lst: float, step_hours: int,
                   max_offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell UTC offset (hours after 00 UTC) for the target local time, and usability mask."""
    offset, dev = nearest_utc_offset(cells.lon, target_lst, step_hours)
    return offset, dev <= max_offset + 1e-9


def lead_hours(offset: np.ndarray, lead_days: np.ndarray) -> np.ndarray:
    """Lead hours, shape (lead_day, cell): 24 (l - 1) + offset."""
    return 24 * (lead_days[:, None] - 1) + offset[None, :]


def init_times(store: Store, year: int, hour: int) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(open_dataset(store).time.values)
    return times[(times.year == year) & (times.hour == hour)]


def read_forecast(store: Store, var: str, inits: pd.DatetimeIndex, hours: np.ndarray, cells: Cells,
                  level: int | None = None) -> np.ndarray:
    """Forecast values at (init, lead_day, cell) for per-cell lead ``hours`` (lead_day, cell)."""
    ds = open_dataset(store)
    times = pd.DatetimeIndex(ds.time.values)
    t_idx = times.get_indexer(inits)
    if (t_idx < 0).any():
        raise KeyError(f"{int((t_idx < 0).sum())} initialisations missing from {store.path}")
    leads = ds.prediction_timedelta.values.astype("timedelta64[h]").astype(int)
    lead_pos = {h: i for i, h in enumerate(leads)}
    needed = np.unique(hours)
    missing = [h for h in needed if h not in lead_pos]
    if missing:
        raise KeyError(f"lead hours {missing} not available in {store.path}")
    l_idx = np.vectorize(lead_pos.get)(hours)
    used = np.array(sorted(lead_pos[h] for h in needed))
    selection = {"time": t_idx, "prediction_timedelta": used}
    if level is not None:
        selection["level"] = [int(np.flatnonzero(ds.level.values == level)[0])]
    require(store, var, selection)
    da = ds[var]
    if level is not None:
        da = da.sel(level=level)
    if da.dims != ("time", "prediction_timedelta", "longitude", "latitude"):
        raise ValueError(f"unexpected dims {da.dims} in {store.path}:{var}")
    remap = np.searchsorted(used, l_idx)  # positions within the loaded lead subset
    out = np.empty((inits.size, hours.shape[0], cells.size), dtype=np.float32)
    col = np.broadcast_to(np.arange(cells.size), hours.shape)
    for i, t in enumerate(t_idx):
        field = da.isel(time=int(t), prediction_timedelta=used).values  # (lead, lon, lat)
        out[i] = field[remap, cells.ilon[col], cells.ilat[col]]
        if (i + 1) % 60 == 0:
            log.info("%s %s: %d/%d initialisations", store.path.split("/")[-2], var, i + 1,
                     inits.size)
    return out


def read_truth(store: Store, var: str, valid: np.ndarray, cells: Cells,
               level: int | None = None) -> np.ndarray:
    """Analysis values at per-element valid times ``valid`` (datetime64, shape (..., cell)).

    Valid times outside the store give NaN.
    """
    ds = open_dataset(store)
    times = pd.DatetimeIndex(ds.time.values)
    flat = pd.DatetimeIndex(valid.reshape(-1))
    pos = times.get_indexer(flat)
    present = np.unique(pos[pos >= 0])
    selection: dict = {"time": present}
    da = ds[var]
    if level is not None:
        lev = int(np.flatnonzero(ds.level.values == level)[0])
        selection["level"] = [lev]
        da = da.isel(level=lev)
    require(store, var, selection)
    block = np.empty((present.size, cells.size), dtype=np.float32)
    for start in range(0, present.size, 400):
        sl = present[start:start + 400]
        block[start:start + sl.size] = cells.take(da.isel(time=sl).values)
    lookup = np.full(times.size, -1)
    lookup[present] = np.arange(present.size)
    rows = np.where(pos >= 0, lookup[np.maximum(pos, 0)], -1).reshape(valid.shape)
    col = np.broadcast_to(np.arange(cells.size), valid.shape)
    out = block[np.maximum(rows, 0), col]
    out[rows < 0] = np.nan
    return out


def build_panels(cfg: Config) -> None:
    cov = xr.open_dataset(cfg.processed / "covariates.nc")
    cells = cells_from_covariates(cov)
    l0, l1 = cfg["lead_days"]
    lead_days = np.arange(l0, l1 + 1)
    max_off = cfg["max_hour_offset"]
    for name, pair in cfg["pairs"].items():
        step = pair["step_hours"]
        aft, aft_ok = target_offsets(cells, cfg["afternoon_lst"], step, max_off)
        night, night_ok = target_offsets(cells, cfg["night_lst"], step, max_off)
        inits = init_times(cfg.store(pair["ai"]), pair["year"], cfg["init_hour_utc"])
        ifs_inits = init_times(cfg.store(pair["ifs"]), pair["year"], cfg["init_hour_utc"])
        if not inits.equals(ifs_inits):
            raise ValueError(f"{name}: AI and IFS initialisations differ")
        h_aft, h_night = lead_hours(aft, lead_days), lead_hours(night, lead_days)
        valid_aft = inits.values[:, None, None] + h_aft[None].astype("timedelta64[h]")
        valid_night = inits.values[:, None, None] + h_night[None].astype("timedelta64[h]")
        data = {}
        for role in ("ai", "ifs"):
            store = cfg.store(pair[role])
            log.info("%s: reading %s afternoon", name, role)
            data[f"{role}_afternoon"] = read_forecast(store, "2m_temperature", inits, h_aft, cells)
            log.info("%s: reading %s night", name, role)
            data[f"{role}_night"] = read_forecast(store, "2m_temperature", inits, h_night, cells)
        for role in ("truth", "alt_truth"):
            store = cfg.store(pair[role])
            data[f"{role}_afternoon"] = read_truth(store, "2m_temperature", valid_aft, cells)
            data[f"{role}_night"] = read_truth(store, "2m_temperature", valid_night, cells)
        dims = ("init", "lead_day", "cell")
        ds = xr.Dataset({k: (dims, v) for k, v in data.items()},
                        coords={"init": inits.values, "lead_day": lead_days})
        ds["afternoon_usable"] = ("cell", aft_ok)
        ds["night_usable"] = ("cell", night_ok)
        ds["afternoon_offset_h"] = ("cell", aft.astype(np.int8))

        if name == "fgn2022":  # 500 hPa geopotential height at lead day 5 (S6)
            five = np.array([5])
            h5 = lead_hours(aft, five)
            v5 = inits.values[:, None, None] + h5[None].astype("timedelta64[h]")
            for role in ("ai", "ifs"):
                z = read_forecast(cfg.store(pair[role]), "geopotential", inits, h5, cells,
                                  level=500)
                ds[f"{role}_z500_day5"] = (("init", "cell"), z[:, 0] / G)
            zt = read_truth(cfg.store(pair["truth"]), "geopotential", v5, cells, level=500)
            ds["truth_z500_day5"] = (("init", "cell"), zt[:, 0] / G)

        out = cfg.processed / f"panel_{name}.nc"
        ds.to_netcdf(out)
        log.info("wrote %s", out)
