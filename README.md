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

## Result

**Both AI ensembles are too cold when soils start dry; the physics ensemble is not.** Over
water-limited land in the warm season, a root zone one standard deviation drier than normal at
initialisation makes afternoon 2 m temperature forecasts at lead days 3–10 too cold by
**0.19 K for FGN** (2022; 95% CI 0.16–0.23) and **0.21 K for GenCast** (2020; 0.15–0.27). IFS ENS
errors move slightly the other way (+0.09 and +0.04 K). All four registered confirmatory tests are
supported after Holm correction. Each test can detect effects of 0.05–0.08 K per SD with 80% power,
smaller than the registered smallest effect of interest (0.10 K per SD).

![Error slope on initial soil moisture by lead day](results/figures/fig1_lead_profile.png)

| Registered test | Estimate (K per SD) | 95% CI | Holm p | Verdict |
|---|---|---|---|---|
| H1: FGN errors depend on soil moisture (β̄ < 0) | −0.191 | [−0.229, −0.157] | < 0.001 | supported |
| H2: more than IFS ENS errors do (Δβ̄ < 0) | −0.280 | [−0.320, −0.238] | < 0.001 | supported |
| H3: GenCast errors depend on soil moisture | −0.213 | [−0.274, −0.150] | < 0.001 | supported |
| H4: more than IFS ENS errors do | −0.248 | [−0.302, −0.198] | < 0.001 | supported |

**The effect has the fingerprints of land memory** (all registered in advance):

- *It sits where soils control evaporation.* FGN's slope is −0.19 in water-limited cells and −0.01 in
  energy-limited cells, and it steepens almost monotonically with coupling strength, from 0.00 in
  the least coupled decile to −0.22 in the most coupled. IFS ENS is flat across deciles.
  ([Figure 2](results/figures/fig2_coupling_deciles.png); map of the regimes in
  [Figure 4](results/figures/fig4_coupling_map.png).)
- *It is an afternoon effect.* The FGN − IFS gap is −0.28 K per SD in the afternoon and −0.17 at
  night (difference −0.12, 95% CI −0.14 to −0.09), as expected when surface fluxes act in daytime.
- *It builds after initialisation.* The gap is larger at lead days 7–10 than at days 1–2, by 0.09 K
  (FGN) and 0.06 K (GenCast).
- *It disappears with the wrong year's soil moisture.* Using soil moisture from each of 30 other
  years (1990–2019) gives slopes between −0.06 and +0.06 (mean 0.00); the true 2022 value of −0.19
  lies below all 30 ([Figure 3](results/figures/fig3_placebo.png)). This rules out fixed geography or
  seasonality as the source.
- *It is not a circulation error.* Controlling for the day-5 500 hPa forecast error leaves 92% of the
  slope (95% CI 87–99%).
- *It survives the registered robustness checks:*
  - swapping the verifying analysis: −0.21 for FGN against ERA5, −0.20 for GenCast against HRES-t0;
  - dropping each continent in turn: −0.17 to −0.21 for FGN;
  - 14-day bootstrap blocks;
  - Driscoll–Kraay standard errors: 0.019, against the bootstrap's 0.019.

**Hot afternoons.** Over dry soils, FGN gives a hot afternoon (above the local 90th percentile) a
9.5% chance at lead day 5, but such afternoons happen 12.5% of the time. Over wet soils it is 4.2%
against 3.6%. So FGN's hot-day probabilities respond too little to the state of the land
([Figure 5](results/figures/fig5_hot_day_probability.png)). IFS ENS over-forecasts hot afternoons in
both strata (20.5% and 8.9%), so its higher probability over dry soils is not better calibration.
FGN's spread–skill ratio is slightly lower over dry soils (0.90 against 0.94). Its CRPS advantage
over IFS ENS is the same over dry and wet soils, so the registered prediction of a larger CRPS cost
over dry soils (S8) is **not supported**.

**What does not fit cleanly.**

- *Within-month changes in soil moisture.* When the slope is identified only from these changes
  (cell × calendar-month fixed effects, check R2), the AI-only slope halves: FGN −0.09 [−0.14,
  −0.04]; GenCast −0.06 [−0.13, +0.02]. The AI − IFS gap widens to about −0.35, because IFS ENS errors
  turn strongly positive (+0.25 to +0.30). The AI cold bias over dry soils is clearest for persistent,
  season-scale anomalies. For short-lived wetting and drying, IFS ENS has its own, larger soil-moisture
  error, in the opposite direction.
- *Exploratory decomposition (not pre-registered).* This splits each error slope into the slope of
  the truth and the slope of the forecast.
  - In 2022, afternoon temperature falls by 0.20 K per SD of wetter soil, IFS ENS forecasts fall by
    0.29 K, and FGN forecasts by 0.01 K: FGN's forecasts essentially ignore the land.
  - In 2020, the truth slope is near zero (+0.04), IFS ENS matches it (0.00), and GenCast's forecasts
    move the wrong way (+0.25).

  These level slopes are observational associations that weather can confound, not causal effects of
  soil moisture. The registered error-slope tests do not need that causal reading: an efficient
  forecast's errors are unpredictable from any issue-time information, whatever its causal role.

All numbers come from [`results/summary.md`](results/summary.md), which `make analyze` regenerates
from [`results/results.json`](results/results.json).

**What it means.** At 1.5° and lead times of 3–10 days, two state-of-the-art AI ensembles leave a
systematic, predictable error that tracks a variable they never see. That error sits where, and
when, land–atmosphere coupling theory says soil moisture matters. The test itself is generic: any
issue-time variable outside a model's inputs (snow, vegetation, irrigation) can be checked the same
way, before anyone retrains a model.

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
results/                  results.json, summary.md, tables/ and figures/, committed with the
                          code commit that produced them (recorded in results.json)
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
