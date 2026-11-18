# Data

All inputs are read from the public [WeatherBench 2](https://weatherbench2.readthedocs.io) bucket
(`gs://weatherbench2/datasets/`, served over HTTPS at `https://storage.googleapis.com/weatherbench2`)
on the 1.5° (240 × 121) grid, conservatively regridded by WeatherBench 2 from 0.25°. Nothing is
redistributed here. `make fetch` mirrors only the chunks the analysis uses into `data/mirror/`,
verifies each one against the MD5 reported by Cloud Storage, and records it in
`data/manifests/*.jsonl`.

| Store | Variables used | Period | Purpose |
|---|---|---|---|
| `era5_daily/1959-2023_01_10-1h-240x121…` | soil water layers 1–3, latent heat flux, daily max 2 m temperature | 1990–2022 | soil-moisture covariate, coupling index, warm season, control *a*₁ |
| `era5/1959-2023_01_10-6h-240x121…` | 2 m temperature, land–sea mask | 1990–2023-01-10 | truth for 2020, control *a*₂, hot-day thresholds |
| `hres_t0/2016-2022-6h-240x121…` | 2 m temperature, geopotential | 2020–2022 | truth for 2022, Z500 check |
| `fgn/2022-240x121…_mean` and members | 2 m temperature, geopotential | 2022, 00 UTC | AI ensemble (WeatherNext 2) |
| `gencast/2020-240x121…_mean` | 2 m temperature | 2020, 00 UTC | AI ensemble (replication) |
| `ifs_ens/2018-2022-240x121…_mean` and members | 2 m temperature, geopotential | 2020, 2022, 00 UTC | physics control |

Exact paths are in [`config/analysis.yaml`](../config/analysis.yaml).

## Licences

Each WeatherBench 2 dataset folder carries its own `LICENSE` file, which governs reuse. ERA5 is
produced by ECMWF for the Copernicus Climate Change Service; IFS ENS and HRES analyses are ECMWF
products; GenCast and FGN forecasts were produced by Google DeepMind. Consult the `LICENSE` file in
each folder before redistributing any derived data.

## Model inputs (why the test makes sense)

From the input tables of the model papers (GenCast: Price et al., 2025, Table B1; FGN: Alet et al.,
2025, Table A.1): six atmospheric variables on 13 pressure levels; 2 m temperature, 10 m winds,
mean sea-level pressure and sea-surface temperature; surface geopotential, land–sea mask, latitude,
longitude, local time and year progress. Neither model receives soil moisture, soil temperature,
snow or vegetation. GenCast conditions on the states at *t* and *t* − 12 h; FGN on two states 6 h
apart. FGN was fine-tuned on, and is initialised from, operational HRES analyses; GenCast was trained
on and is initialised from ERA5.
