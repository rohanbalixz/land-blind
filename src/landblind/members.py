"""Member-level scores at lead day 5 (preregistration S7–S9), FGN and IFS ENS, 2022."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from landblind.climatology import noleap_doy
from landblind.config import Config
from landblind.grid import Cells
from landblind.mirror import Store, open_dataset, require
from landblind.panels import cells_from_covariates, lead_hours, read_truth, target_offsets
from landblind.scores import crps_fair, ensemble_moments, exceedance_probability

log = logging.getLogger(__name__)

MEMBER_DIM = {"fgn_members": "sample", "ifs_members": "number"}


def member_values(store: Store, member_dim: str, inits: pd.DatetimeIndex, hours: np.ndarray,
                  cells: Cells):
    """Yield (init position, members (m, cell)) at per-cell lead ``hours`` (cell,)."""
    ds = open_dataset(store)
    times = pd.DatetimeIndex(ds.time.values)
    t_idx = times.get_indexer(inits)
    leads = ds.prediction_timedelta.values.astype("timedelta64[h]").astype(int)
    pos = {h: i for i, h in enumerate(leads)}
    used = np.array(sorted({pos[int(h)] for h in np.unique(hours)}))
    remap = np.searchsorted(used, np.array([pos[int(h)] for h in hours]))
    require(store, "2m_temperature", {"time": t_idx, "prediction_timedelta": used})
    da = ds["2m_temperature"].transpose("time", member_dim, "prediction_timedelta", "longitude",
                                        "latitude")
    for i, t in enumerate(t_idx):
        field = da.isel(time=int(t), prediction_timedelta=used).values  # (m, lead, lon, lat)
        yield i, field[:, remap, cells.ilon, cells.ilat]


def build_member_scores(cfg: Config) -> None:
    cov = xr.open_dataset(cfg.processed / "covariates.nc")
    cells = cells_from_covariates(cov)
    lead_day = cfg["members"]["lead_day"]
    aft, _ = target_offsets(cells, cfg["afternoon_lst"], 6, cfg["max_hour_offset"])
    hours = lead_hours(aft, np.array([lead_day]))[0]
    every = cfg["members"]["init_every_n_days"]

    base = pd.DatetimeIndex(open_dataset(cfg.store("fgn_members")).time.values)
    inits = base[(base.year == 2022) & (base.hour == cfg["init_hour_utc"])
                 & (((base.day_of_year - 1) % every) == 0)]
    valid = inits.values[:, None] + hours[None].astype("timedelta64[h]")
    truth = read_truth(cfg.store("hres_t0"), "2m_temperature", valid, cells)
    doy = noleap_doy(pd.DatetimeIndex(valid.reshape(-1))).reshape(valid.shape)
    q90 = cov.q90_afternoon.values[doy - 1, np.arange(cells.size)[None, :]]

    attrs = {"lead_day": lead_day}
    data = {"truth": truth, "q90": q90, "obs_hot": (truth > q90).astype(np.float32)}
    data["obs_hot"][~np.isfinite(truth)] = np.nan
    for name in ("fgn_members", "ifs_members"):
        tag = name.split("_")[0]
        shape = (inits.size, cells.size)
        mean, var, crps, prob = (np.full(shape, np.nan, dtype=np.float32) for _ in range(4))
        for i, x in member_values(cfg.store(name), MEMBER_DIM[name], inits, hours, cells):
            mu, v = ensemble_moments(x, axis=0)
            mean[i], var[i] = mu, v
            crps[i] = crps_fair(x, truth[i], axis=0)
            prob[i] = exceedance_probability(x, q90[i], axis=0)
            if (i + 1) % 30 == 0:
                log.info("%s: %d/%d initialisations", name, i + 1, inits.size)
        attrs[f"{tag}_n_members"] = int(x.shape[0])
        data.update({f"{tag}_mean": mean, f"{tag}_var": var, f"{tag}_crps": crps,
                     f"{tag}_prob_hot": prob})
        log.info("%s: %d members", name, x.shape[0])
    ds = xr.Dataset({k: (("init", "cell"), v) for k, v in data.items()},
                    coords={"init": inits.values},
                    attrs=attrs)
    out = cfg.processed / "members_fgn2022.nc"
    ds.to_netcdf(out)
    log.info("wrote %s", out)
