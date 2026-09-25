# Validation summary

**Status:** preliminary Validation-only comparison. It is not a Final OOS result and must not be used as a trading recommendation.

## Configuration

- Target dates: 2022-01-03 to 2023-12-29 (497 forecasts).
- Estimation: 756-trading-day rolling window, refit every 5 sessions.
- Models: GARCH(1,1)-t, EGARCH(1,1)-t, GJR-GARCH(1,1)-t, and log-HAR-RV.
- Target: next quality-approved session's intraday realized variance. QLIKE is the primary selection loss; lower is better.

## Results

| Model | QLIKE | MAE | RMSE |
|---|---:|---:|---:|
| HAR-RV | -8.627343 | 0.00003469 | 0.00005923 |
| GJR-GARCH | -8.594002 | 0.00004454 | 0.00006676 |
| GARCH | -8.574015 | 0.00005012 | 0.00007650 |
| EGARCH | -8.567351 | 0.00005023 | 0.00007283 |

The lowest Validation QLIKE is **HAR-RV** (-8.627343). This is a candidate, not a locked final model, because alternative windows have not yet been compared.

## QLIKE Diebold–Mariano comparisons involving the current leader

| Comparison | DM statistic | Two-sided p-value |
|---|---:|---:|
| EGARCH vs HAR-RV | 5.0531 | 4.34739e-07 |
| GARCH vs HAR-RV | 4.7304 | 2.24115e-06 |
| GJR-GARCH vs HAR-RV | 3.2708 | 0.00107235 |

A positive statistic means the first listed model has a higher QLIKE loss than the second.

## Protected final holdout

The Final OOS target interval (2024-01-02 to 2026-09-18; 674 dates) is present only in the split manifest and has not been forecast, scored, or used to choose this configuration.
