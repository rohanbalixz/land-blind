# Deviations from the pre-registration

Every change to the registered plan ([`preregistration.md`](preregistration.md) and
[`config/analysis.yaml`](../config/analysis.yaml)) is recorded here, with the date, what changed, why,
and whether the change was made before or after the affected data were read.

| # | Date | Section | Change | Reason | Data seen? |
|---|---|---|---|---|---|
| 1 | 2026-11-18 | §4.2 | Clarification. The day-of-year standard deviation is computed from anomalies about each day's window mean, not from raw values in the window. | A unit test showed that the raw-value version absorbs the slope of the seasonal cycle inside the ±15-day window (an SD of 2.0 was estimated as 2.28 where the cycle is steep). The anomaly version is what "standardised anomaly" is meant to be. | No. Made before any data values were read. |
