# Deviations from the pre-registration

Every change to the registered plan ([`preregistration.md`](preregistration.md) and
[`config/analysis.yaml`](../config/analysis.yaml)) is recorded here, with the date, what changed, why,
and whether the change was made before or after the affected data were read.

| # | Date | Section | Change | Reason | Data seen? |
|---|---|---|---|---|---|
| 1 | 2026-11-18 | §4.2 | Clarification. The day-of-year standard deviation is computed from anomalies about each day's window mean, not from raw values in the window. | A unit test showed that the raw-value version absorbs the slope of the seasonal cycle inside the ±15-day window (an SD of 2.0 was estimated as 2.28 where the cycle is steep). The anomaly version is what "standardised anomaly" is meant to be. | No. Made before any data values were read. |

## Changes made after the confirmatory results were seen

None of these alters a registered test, threshold or sample. They are listed so a reader can see
exactly what was added once the results were known.

| # | Date | Change | Effect on registered results |
|---|---|---|---|
| 2 | 2026-11-18 | Download layer: truncated HTTP reads are retried and failed chunks are collected, after an `IncompleteRead` stopped the first download. | None (data acquisition only). |
| 3 | 2026-11-18 | Fixed-effects demeaning vectorised with sparse group matrices (10.5 min → 2.8 min). | None: every number in `results.json` agrees with the first run to 1.8 × 10⁻¹⁴. |
| 4 | 2026-11-18 | **Exploratory, not pre-registered:** slopes of the truth and of each forecast on *s* (`exploratory_level_slopes`), under the registered and the cell × month fixed effects. | None. Added to interpret the error slopes. These are observational associations, not causal effects of soil moisture. |
| 5 | 2026-11-18 | S3–S5 are also reported for GenCast. The plan registered them for FGN only; the GenCast rows are labelled "not registered". | None. |
| 6 | 2026-11-18 | Figure 5 (hot-day probabilities by soil-moisture stratum) added. | None (presentation of S7). |
