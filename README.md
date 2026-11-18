# land-memory

**Do AI weather ensembles miss the soil moisture they never see?**

GenCast and FGN (the model behind WeatherNext 2) forecast the atmosphere from the atmosphere alone.
Their inputs contain no soil moisture, soil temperature, snow or vegetation. The physics-based IFS
ensemble carries a land-surface model initialised from a land data assimilation. A forecast that
uses everything known at issue time has errors that nothing known at issue time can predict. So if
the AI ensembles' 2 m temperature errors can be predicted from the initial soil-moisture anomaly,
they are leaving land memory on the table, and the cost should fall where land controls heat:
water-limited regions, in the afternoon, at longer leads.

This repository tests that, pre-registered, on every land grid cell, for two AI ensembles in two
years, with IFS ENS as the physics control.

<!-- RESULTS -->

## Design

| | |
|---|---|
| **Systems** | FGN / WeatherNext 2 (64 members, 2022) and GenCast (56 members, 2020) against IFS ENS (50 members, same years), from WeatherBench 2 at 1.5° |
| **Outcome** | Afternoon (≈15:00 local solar time) 2 m temperature error of the ensemble mean, lead days 1–15, 00 UTC initialisations, each cell's three warmest months |
| **Covariate** | ERA5 root-zone (0–100 cm) soil-moisture anomaly on the day before initialisation, standardised by its 1990–2019 day-of-year climatology |
| **Where** | Water-limited cells: top tercile of the terrestrial coupling index (Dirmeyer, 2011) from ERA5 1990–2019 fluxes, computed before any forecast was opened |
| **Estimand** | β̄: slope of the error on the covariate, K per standard deviation, averaged over lead days 3–10; cell and date fixed effects; controls for the previous day's maximum temperature anomaly and the initial-time temperature anomaly, which the AI models do see |
| **Contrast** | Δβ̄ = β̄<sub>AI</sub> − β̄<sub>IFS</sub>. The verifying analysis cancels, so Δβ̄ does not depend on the choice of truth |
| **Inference** | 7-day moving-block bootstrap over initialisation dates (whole maps resampled together, B = 2000), Driscoll–Kraay errors as a cross-check, Holm across the four confirmatory tests, equivalence bounds of ±0.10 K per SD so that a null is informative |
| **Fingerprints** | growth with lead time; afternoon versus night; water- versus energy-limited cells; dose–response across coupling deciles |
| **Negative controls** | the same analysis with soil moisture from each of 30 wrong years; a 500 hPa circulation-error control |
| **Extremes** | hot-day (local 90th percentile) probabilities, fair CRPS and spread–skill from the ensemble members, stratified by initial soil moisture, which is known before the forecast so the stratification does not condition on the outcome |

The full plan, with every threshold and the reading of each possible result fixed in advance, is in
[`docs/preregistration.md`](docs/preregistration.md). It was committed as `00f4f2c` on 2026-11-18,
before any forecast, analysis or reanalysis values were read. Every departure from it is logged in
[`docs/deviations.md`](docs/deviations.md).

## Reproduce

```bash
conda env create -f environment.yml
conda activate landmem
make test      # unit tests and a synthetic end-to-end run with a planted effect
make all       # download (~30 GB), covariates, panels, member scores, tests, figures
```

`make all` mirrors only the needed chunks of the public WeatherBench 2 bucket into `data/`,
checking each against the MD5 that Cloud Storage reports, and resumes if interrupted. Results land
in `results/results.json`, `results/summary.md`, `results/tables/` and `results/figures/`.

## Layout

```
config/analysis.yaml      every registered parameter
docs/preregistration.md   hypotheses, estimands, decision rules (registered first)
docs/deviations.md        dated log of every change to the plan
docs/data.md              data sources, model inputs and licences
src/landmem/
  mirror.py               partial, checksum-verified mirror of public Zarr stores
  fetch.py                the registered data selection
  covariates.py           ERA5-only: domain, soil-moisture anomalies, coupling, warm season
  panels.py               afternoon/night forecast and truth arrays per model pair
  members.py, scores.py   fair CRPS, spread and hot-day probabilities from members
  stats.py                fixed-effects slopes, block bootstrap, Driscoll–Kraay, Holm, TOST
  analysis.py, report.py  the registered tests and the generated summary
  figures.py
tests/                    unit tests and the synthetic end-to-end test
results/                  outputs committed with the code that produced them
```

## Limitations

- Truth is gridded analysis (HRES-t0 for 2022, ERA5 for 2020), not station data, at 1.5°.
- ERA5 soil moisture comes from the same family of land model as IFS ENS. If its anomalies match the
  IFS initial state more closely than reality does, the AI−IFS contrast could favour IFS ENS. The
  AI-only tests (H1, H3) are not affected.
- Each AI model is evaluated in one year; spatial pooling and the block bootstrap address precision,
  not year-specific weather.
- 2022 was FGN's validation year. If it lies inside the training data used for these forecasts,
  memorisation can only shrink FGN's errors, which makes its test conservative.
- GenCast steps every 12 h, so its afternoon sample is limited to longitudes where 00 or 12 UTC falls
  within 3 h of 15:00 local time.

## Citation

See [`CITATION.cff`](CITATION.cff). Code is MIT-licensed; data sources and their licences are listed
in [`docs/data.md`](docs/data.md).
