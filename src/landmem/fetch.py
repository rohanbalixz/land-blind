"""What to download: the registered selection of every store, expressed as chunk requests."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from landmem.config import Config
from landmem.mirror import Store, fetch_metadata, fetch_selection, open_dataset

log = logging.getLogger(__name__)

ROOT_ZONE = ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
             "volumetric_soil_water_layer_3"]
ERA5_DAILY_VARS = [*ROOT_ZONE, "mean_surface_latent_heat_flux", "2m_temperature_max"]


def time_index(store: Store, start: str, end: str, hours: tuple[int, ...] | None = None,
               every_n_days: int = 1) -> np.ndarray:
    """Integer positions along ``time`` between ``start`` and ``end`` (inclusive)."""
    times = pd.DatetimeIndex(open_dataset(store)["time"].values)
    keep = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    if hours is not None:
        keep &= np.isin(times.hour, hours)
    if every_n_days > 1:
        keep &= ((times.day_of_year - 1) % every_n_days) == 0
    return np.flatnonzero(keep)


def lead_index(store: Store, hours: np.ndarray) -> np.ndarray:
    """Integer positions along ``prediction_timedelta`` for the given lead hours."""
    leads = open_dataset(store)["prediction_timedelta"].values.astype("timedelta64[h]").astype(int)
    pos = np.flatnonzero(np.isin(leads, hours))
    if pos.size != np.unique(hours).size:
        raise ValueError(f"lead hours {sorted(set(hours) - set(leads))} not in {store.path}")
    return pos


def day5_afternoon_hours(step_hours: int) -> np.ndarray:
    """Lead hours that can be the lead-day-5 afternoon target for some longitude."""
    offsets = np.arange(step_hours, 25, step_hours)
    return 24 * 4 + offsets


def fetch_all_metadata(cfg: Config) -> None:
    for name in cfg["stores"]:
        log.info("metadata: %s", name)
        fetch_metadata(cfg.store(name))


def fetch_era5(cfg: Config, workers: int) -> None:
    y0, _ = cfg["climatology_years"]
    daily = cfg.store("era5_daily")
    fetch_selection(daily, ERA5_DAILY_VARS,
                    {"time": time_index(daily, f"{y0}-01-01", "2022-12-31")}, workers=workers)
    six = cfg.store("era5_6h")
    fetch_selection(six, ["2m_temperature"],
                    {"time": time_index(six, f"{y0}-01-01", "2023-01-10T18")}, workers=workers)


def fetch_truth(cfg: Config, workers: int) -> None:
    hres = cfg.store("hres_t0")
    fetch_selection(hres, ["2m_temperature"],
                    {"time": time_index(hres, "2020-01-01", "2022-12-31T18")}, workers=workers)
    fetch_selection(hres, ["geopotential"],
                    {"time": time_index(hres, "2022-01-01", "2022-12-31T18")}, workers=workers)


def fetch_forecast_means(cfg: Config, workers: int) -> None:
    hour = cfg["init_hour_utc"]
    for pair in cfg["pairs"].values():
        year = pair["year"]
        for name in (pair["ai"], pair["ifs"]):
            store = cfg.store(name)
            inits = time_index(store, f"{year}-01-01", f"{year}-12-31T23", hours=(hour,))
            fetch_selection(store, ["2m_temperature"], {"time": inits}, workers=workers)
    # Lead-day-5 geopotential for the circulation check (S6), FGN pair only.
    pair = cfg["pairs"]["fgn2022"]
    for name in (pair["ai"], pair["ifs"]):
        store = cfg.store(name)
        inits = time_index(store, "2022-01-01", "2022-12-31T23", hours=(hour,))
        leads = lead_index(store, day5_afternoon_hours(pair["step_hours"]))
        fetch_selection(store, ["geopotential"], {"time": inits, "prediction_timedelta": leads},
                        workers=workers)


def fetch_members(cfg: Config, workers: int) -> None:
    every = cfg["members"]["init_every_n_days"]
    hour = cfg["init_hour_utc"]
    for name in ("fgn_members", "ifs_members"):
        store = cfg.store(name)
        inits = time_index(store, "2022-01-01", "2022-12-31T23", hours=(hour,), every_n_days=every)
        leads = lead_index(store, day5_afternoon_hours(6))
        fetch_selection(store, ["2m_temperature"], {"time": inits, "prediction_timedelta": leads},
                        workers=workers)


STEPS = {
    "metadata": lambda cfg, w: fetch_all_metadata(cfg),
    "era5": fetch_era5,
    "truth": fetch_truth,
    "forecasts": fetch_forecast_means,
    "members": fetch_members,
}
