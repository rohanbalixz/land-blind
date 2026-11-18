# Pre-registration: soil moisture the model cannot see

| | |
|---|---|
| **Registered** | 2026-11-18, before any forecast, analysis or reanalysis values were read |
| **Author** | Rohan Bali |
| **Parameters** | [`config/analysis.yaml`](../config/analysis.yaml) (part of this registration) |
| **Deviations** | logged in [`docs/deviations.md`](deviations.md) with date and reason |

## 1. Question

GenCast (Price et al., 2025) and FGN, the model behind WeatherNext 2 (Alet et al., 2025), take no land-surface state as input. Their input tables list six atmospheric variables on 13 pressure levels, five single-level variables (2 m temperature, 10 m winds, mean sea-level pressure, sea-surface temperature), static fields and clock features: no soil moisture, soil temperature, snow or vegetation. Both condition on the two most recent atmospheric states only (GenCast: *t* and *t* − 12 h; FGN: *t* and *t* − 6 h). The physics-based IFS ensemble (IFS ENS) carries a prognostic soil-moisture state initialised by a land data assimilation.

A forecast that uses all information available when it is issued has errors that cannot be predicted from that information (forecast efficiency: E[*e* | 𝓘<sub>*t*</sub>] = 0). Soil moisture is in the issue-time information set but outside the AI models' inputs. If the atmospheric state carries all of its predictive content for near-surface temperature, AI forecast errors are orthogonal to soil moisture. If it does not, AI errors are predictable from it, and physically they should be most predictable where evapotranspiration is water-limited, in the afternoon, and at longer lead times.

> **Primary question.** Are AI ensemble forecast errors of afternoon 2 m temperature predictable from the initial soil-moisture anomaly that the model never sees, and more so than the errors of IFS ENS?

## 2. State of knowledge at registration

- Only store metadata had been read: variable names, array shapes, chunk layouts, coordinate names and the input tables of the two model papers.
- To measure bandwidth, nine compressed forecast chunks (FGN and IFS ENS 2 m temperature) were streamed to `/dev/null`. They were not decoded or stored.
- No ERA5, HRES or forecast values had been read, and no covariate (soil-moisture anomaly, coupling index, warm season) had been computed.

## 3. Data

All data come from the public WeatherBench 2 bucket (`gs://weatherbench2/datasets/`) on the 1.5° (240 × 121) conservatively regridded grid. Exact store paths are in the configuration file.

| System | Role | Year | Members | Lead step | Initialisations used |
|---|---|---|---|---|---|
| FGN (WeatherNext 2) | AI, no land input | 2022 | 64 | 6 h, 6–360 h | 00 UTC |
| GenCast | AI, no land input | 2020 | 56 | 12 h, 12–360 h | 00 UTC |
| IFS ENS | physics control with land model | 2020, 2022 | 50 | 6 h, 0–360 h | 00 UTC |
| ERA5 | covariates; truth for 2020 | 1990–2022 | – | daily and 6 h | – |
| HRES-t0 | truth for 2022 | 2022 | – | 6 h | – |

Each AI model is verified against the analysis it was trained to predict: FGN was fine-tuned on and initialised from HRES-fc0 (operational analyses), and GenCast was trained on and initialised from ERA5. The other analysis is a pre-registered robustness check (R1). IFS ENS is verified against the same truth as the AI model it is paired with.

## 4. Definitions

**4.1 Domain.** Grid cells with ERA5 land–sea mask ≥ 0.5, latitude between 60° S and 80° N, and a warm-season standard deviation of daily root-zone soil-moisture anomalies (1990–2019) of at least 0.005 m³ m⁻³. The last condition removes ice sheets and hyper-arid cells whose soil moisture does not vary. Every regression and every quantile uses cos(latitude) area weights.

**4.2 Climatology.** 1990–2019. Day-of-year statistics use a ±15-day window (31 days × 30 years). 29 February is treated as 28 February.

**4.3 Soil-moisture covariate.** Root-zone volumetric soil water (0–100 cm) θ = (7·L1 + 21·L2 + 72·L3)/100, from ERA5 daily means of layers 1–3 (0–7, 7–28, 28–100 cm). The covariate *s*(*c*, *D*) is the standardised anomaly of θ at cell *c* on day *D* − 1, which lies entirely before the 00 UTC initialisation on day *D*. Negative *s* means drier than normal.

**4.4 Warm season.** For each cell, the three consecutive calendar months with the highest 1990–2019 mean of ERA5 daily maximum 2 m temperature. An initialisation date is in season if its calendar month is one of the three.

**4.5 Coupling regime.** Terrestrial coupling index (Dirmeyer, 2011) TCI = cov(θ′, LH′) / sd(θ′), where LH is the upward latent heat flux (the negative of ERA5 `mean_surface_latent_heat_flux`) and primes denote anomalies from the day-of-year climatology, using warm-season days 1990–2019. **Water-limited** cells are the top area-weighted tercile of TCI in the domain; **energy-limited** cells are the bottom tercile.

**4.6 Target time and lead day.** Local solar time is LST = UTC + longitude/15 h. The *afternoon* hour of a cell is the 6-hourly UTC hour nearest to 15:00 LST (at most 3 h away). For an initialisation at 00 UTC on day *D*, lead day ℓ ∈ {1, …, 15} targets the valid time *D* + 24(ℓ − 1) h + δ<sub>*c*</sub>, where δ<sub>*c*</sub> ∈ {6, 12, 18, 24} h is the cell's afternoon offset. *Night* uses 03:00 LST in the same way. GenCast is 12-hourly, so δ<sub>*c*</sub> ∈ {12, 24} h and only cells whose afternoon hour is within 3 h of 15:00 LST are used (longitudes 0–90° E and 90–180° W).

**4.7 Error.** *e* = *y* − *ȳ*, where *y* is the verifying analysis at the valid time and *ȳ* is the ensemble mean from the WeatherBench 2 `_mean` store. Positive *e* means the forecast was too cold.

**4.8 Controls.** Both controls are in the AI models' information set: *a*₁ is the ERA5 daily-maximum 2 m temperature anomaly on day *D* − 1 (K), and *a*₂ is the ERA5 2 m temperature anomaly at 00 UTC on day *D* (K, hour-matched climatology). They separate a land-memory effect from a generic failure on warm states.

**4.9 Sample.** In-season, 00 UTC initialisations at domain cells. A (cell, initialisation) pair is kept only if forecast and truth exist at all 15 lead days, so every lead uses the same sample, and the AI model and IFS ENS use the identical sample.

## 5. Primary analysis

**5.1 Estimator.** For model *M* and lead day ℓ, over water-limited cells:

*e*<sub>*c*,*D*,ℓ</sub> = α<sub>*c*</sub> + τ<sub>*D*</sub> + β<sub>*M*</sub>(ℓ) *s*<sub>*c*,*D*</sub> + γ₁ *a*₁ + γ₂ *a*₂ + ε,

estimated by area-weighted least squares with cell and initialisation-date fixed effects. Fixed effects and controls are partialled out once on the full sample (Frisch–Waugh–Lovell), and inference resamples date-level sufficient statistics.

- **Primary summary:** β̄<sub>*M*</sub> = mean of β<sub>*M*</sub>(ℓ) over ℓ = 3, …, 10, in K per standard deviation of soil-moisture anomaly. A negative β̄ means the forecast is too cold over dry soils.
- **Contrast:** Δβ̄ = β̄<sub>AI</sub> − β̄<sub>IFS</sub> on the identical sample. The truth cancels (*e*<sub>AI</sub> − *e*<sub>IFS</sub> = *ȳ*<sub>IFS</sub> − *ȳ*<sub>AI</sub>), so Δβ̄ does not depend on the verifying dataset.

**5.2 Inference.** A moving-block bootstrap over initialisation dates: blocks of 7 days, circular within the year, B = 2000 replicates, seed 20261118. All cells and lead days of a date are resampled together, which preserves spatial and cross-lead dependence. Tests use *z* = estimate / bootstrap SE. Intervals are bootstrap percentile intervals.

**5.3 Confirmatory hypotheses.** One-sided, α = 0.05, Holm-adjusted across the four tests:

| | Hypothesis | Truth |
|---|---|---|
| **H1** | β̄<sub>FGN</sub> < 0 | HRES-t0 |
| **H2** | Δβ̄<sub>FGN − IFS</sub> < 0 | none needed |
| **H3** | β̄<sub>GenCast</sub> < 0 | ERA5 |
| **H4** | Δβ̄<sub>GenCast − IFS</sub> < 0 | none needed |

**5.4 Equivalence and power.** The smallest effect size of interest is |β̄| = 0.10 K per standard deviation. At that size a two-standard-deviation drought biases afternoon forecasts by 0.2 K; below it, a systematic land-memory bias would be negligible for users next to ordinary day-to-day forecast error. A null is called *practically orthogonal* when the 90% interval lies inside ±0.10 (two one-sided tests at α = 0.05). The minimum detectable effect at 80% power, MDE = 2.486 × SE, is reported for every test. A test that is neither rejected nor equivalent is reported as **inconclusive**.

**5.5 Interpretation fixed in advance.**

| Result | Reading |
|---|---|
| H1 and H2 supported | The AI ensemble misses land memory that IFS ENS captures. |
| H1 supported, H2 not | Both systems' errors depend on soil moisture: a shared limitation or a verification artefact, not an AI-specific gap. |
| H1 equivalent to zero | The AI ensemble recovers the land state from the atmosphere at 3–10 days. This is an informative null. |
| H1 inconclusive | The data cannot resolve effects of the size that matters. |

The same reading applies to H3 and H4 for GenCast. Agreement between FGN 2022 and GenCast 2020 counts as replication across models and years.

## 6. Secondary analyses (directional, not adjusted for multiplicity)

Each is reported with its estimate and 95% interval.

- **S1 Lead growth.** Δβ averaged over lead days 7–10 is below Δβ averaged over lead days 1–2 (FGN and GenCast).
- **S2 Diurnal fingerprint.** For FGN, Δβ̄ in the afternoon is below Δβ̄ at night, because surface flux partitioning acts in daytime.
- **S3 Regime contrast.** Δβ̄ and β̄<sub>FGN</sub> are lower in water-limited than in energy-limited cells.
- **S4 Dose–response.** Across area-weighted TCI deciles, the area-weighted linear trend of β̄<sub>FGN</sub> against decile rank is negative.
- **S5 Wrong-year placebo.** Replace *s* with the same-calendar-day anomaly from each of the 30 years 1990–2019 and recompute β̄<sub>FGN</sub>. Prediction: the true-year β̄<sub>FGN</sub> lies below at least 95% of the placebo values. Rank *p* = (1 + #{placebo ≤ true}) / 31.
- **S6 Circulation pathway.** At lead day 5, adding the 500 hPa geopotential forecast error (same cell and valid time) as a covariate retains at least half of β<sub>FGN</sub>(5). The slope of the Z500 error on *s* is reported descriptively.
- **S7 Hot-tail reliability.** Uses ensemble members at lead day 5 for every second in-season initialisation. The event is *y* above the hour-matched ERA5 1990–2019 90th percentile (±7-day window). In water-limited cells with *s* < −0.43 (driest tercile of a standard normal), the observed frequency minus the mean FGN probability, ō − p̄<sub>FGN</sub>, is positive, and p̄<sub>IFS</sub> − p̄<sub>FGN</sub> is positive.
- **S8 Proper-score cost.** The CRPS skill of FGN relative to IFS ENS, 1 − CRPS<sub>FGN</sub>/CRPS<sub>IFS</sub> (fair CRPS), is lower in the dry stratum (*s* < −0.43) than in the wet stratum (*s* > 0.43).
- **S9 Spread.** The FGN spread–skill ratio (finite-ensemble corrected) is lower in the dry stratum than in the wet stratum.

S7–S9 stratify on *s*, which is known before the forecast is issued, so they do not condition on the outcome.

## 7. Robustness (reported; does not decide hypotheses)

- **R1 Truth swap:** FGN against ERA5; GenCast against HRES-t0.
- **R2 Fixed effects:** cell × calendar month plus date; date only; no controls.
- **R3 Block length:** 14 days.
- **R4 Layer:** surface layer L1 instead of the root zone.
- **R5 Leave one continent out:** boxes are defined in the configuration file.
- **R6 Analytic standard errors:** Driscoll–Kraay standard errors (Bartlett kernel, 7 lags) instead of the bootstrap.
- **R7 End-to-end recovery:** inject *ȳ*<sub>syn</sub> = *ȳ*<sub>IFS</sub> + κ*s* with κ ∈ {0.05, 0.10, 0.20} K per SD, and confirm the pipeline returns Δβ̄ = −κ.

## 8. Exclusions and missing data

- Initialisations missing from any store are dropped and listed in the run log.
- No outliers are removed and no values are winsorised.

## 9. Limitations stated in advance

- Truth is gridded analysis, not station data, at 1.5° resolution.
- ERA5 soil moisture comes from the same family of land model (H-TESSEL) that IFS ENS uses. If its anomalies match the IFS initial state more closely than reality does, H2 could favour IFS ENS. H1 and H3 are not affected.
- Each AI model is evaluated in one year. Spatial pooling over many cells and the block bootstrap address precision, but not year-specific weather.
- 2022 was the validation year in the FGN paper, and the WeatherBench 2 dataset does not state which checkpoint produced these forecasts. If 2022 is inside FGN's training data, memorisation can only shrink its errors, which makes H1 conservative.
- The two AI models differ in time step, so GenCast's afternoon sample is restricted in longitude (§4.6).

## References

- Alet, F. et al. (2025). Skillful joint probabilistic weather forecasting from marginals. arXiv:2506.10772.
- Dirmeyer, P. A. (2011). The terrestrial segment of soil moisture–climate coupling. *Geophysical Research Letters*, 38, L16702.
- Driscoll, J. C. and Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80, 549–560.
- Fortin, V. et al. (2014). Why should ensemble spread match the RMSE of the ensemble mean? *Journal of Hydrometeorology*, 15, 1708–1713.
- Price, I. et al. (2025). Probabilistic weather forecasting with machine learning. *Nature*, 637, 84–90.
- Rasp, S. et al. (2024). WeatherBench 2: A benchmark for the next generation of data-driven global weather models. *Journal of Advances in Modeling Earth Systems*, 16, e2023MS004019.
