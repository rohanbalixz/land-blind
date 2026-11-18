"""Grid geometry: candidate land cells, area weights, local solar time and continents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cells:
    """A flat set of grid cells taken from a (longitude, latitude) grid."""

    ilon: np.ndarray
    ilat: np.ndarray
    lon: np.ndarray  # degrees east in [0, 360)
    lat: np.ndarray

    @property
    def size(self) -> int:
        return int(self.lon.size)

    @property
    def weight(self) -> np.ndarray:
        """Area weights proportional to cos(latitude)."""
        return np.cos(np.deg2rad(self.lat))

    @property
    def lon180(self) -> np.ndarray:
        return ((self.lon + 180.0) % 360.0) - 180.0

    def take(self, field: np.ndarray) -> np.ndarray:
        """Select the cells from an array whose last two axes are (longitude, latitude)."""
        return field[..., self.ilon, self.ilat]

    def subset(self, mask: np.ndarray) -> Cells:
        return Cells(self.ilon[mask], self.ilat[mask], self.lon[mask], self.lat[mask])


def candidate_land_cells(lon: np.ndarray, lat: np.ndarray, land_sea_mask: np.ndarray,
                         lsm_min: float, lat_min: float, lat_max: float) -> Cells:
    """Cells with land fraction >= ``lsm_min`` inside the latitude band.

    ``land_sea_mask`` has shape (longitude, latitude), as in the WeatherBench 2 stores.
    """
    _, lat2 = np.meshgrid(lon, lat, indexing="ij")
    keep = (land_sea_mask >= lsm_min) & (lat2 >= lat_min) & (lat2 <= lat_max)
    ilon, ilat = np.nonzero(keep)
    return Cells(ilon, ilat, lon[ilon] % 360.0, lat[ilat])


def local_solar_hour(utc_hour: np.ndarray | float, lon: np.ndarray) -> np.ndarray:
    """Local solar time (hours in [0, 24)) at longitude ``lon`` (degrees east)."""
    return (np.asarray(utc_hour, dtype=float) + np.asarray(lon, dtype=float) / 15.0) % 24.0


def nearest_utc_offset(lon: np.ndarray, target_lst: float, step_hours: int) -> tuple[np.ndarray,
                                                                                  np.ndarray]:
    """UTC offset from 00 UTC (in hours, one of ``step, 2*step, ..., 24``) whose local solar time
    is nearest to ``target_lst``, and the absolute deviation in hours.

    An offset of 24 h stands for 00 UTC of the next day, so lead day 1 covers (0, 24] h.
    Ties are broken towards the earlier offset.
    """
    offsets = np.arange(step_hours, 25, step_hours, dtype=float)  # e.g. 6, 12, 18, 24
    lst = local_solar_hour(offsets[:, None] % 24.0, np.asarray(lon)[None, :])
    dev = np.abs(((lst - target_lst) + 12.0) % 24.0 - 12.0)
    best = np.argmin(dev + 1e-9 * np.arange(offsets.size)[:, None], axis=0)
    return offsets[best].astype(int), dev[best, np.arange(dev.shape[1])]


def assign_continents(cells: Cells, boxes: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Name of the first box containing each cell ('other' if none)."""
    names = np.full(cells.size, "other", dtype=object)
    lon = cells.lon180
    for name, (lon0, lon1, lat0, lat1) in boxes.items():
        inside = (lon >= lon0) & (lon <= lon1) & (cells.lat >= lat0) & (cells.lat <= lat1)
        names[(names == "other") & inside] = name
    return names.astype(str)
