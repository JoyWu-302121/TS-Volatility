# SPX One-Minute Volatility Extension

This directory is the repository's self-contained SPX one-minute realized-volatility reconstruction and forecasting extension. It is isolated from the legacy notebooks and canonical 30-minute realized-variance workflow, and does not make a trading recommendation.

It will rebuild SPX realized variance from one-minute close-to-close log returns and evaluate volatility forecasts with point-in-time data only.

## Structure

- `code/`: exploratory and reporting notebooks.
- `scripts/`: reproducible data-building and model-running scripts.
- `data/raw/SPX_1min.csv`: versioned one-minute SPX OHLC source data.
- `data/processed/`: derived one-minute returns, session-quality diagnostics, and realized-variance datasets.
- `requirements.txt`: dependencies specific to this extension.

## Raw-data contract

`SPX_1min.csv` contains `DateTime`, `Symbol`, `Open`, `High`, `Low`, `Close`, and `Volume`. It spans 2010-01-04 through 2026-09-18. The `Volume` field is zero throughout, so it must not be used for liquidity, VWAP, or volume-weighted-volatility analysis.

The reconstruction must sort timestamps, calculate log returns only within each trading day, exclude overnight returns, preserve the number of valid intraday intervals, and flag incomplete or non-standard sessions before estimating daily realized variance.

## Initial forecasting design

The initial experiment forecasts the next quality-approved trading session's realized variance. Daily RV is the sum of squared one-minute close-to-close log returns within the regular 09:30–16:00 session. A session is model-eligible only when it has at least 300 intraday returns and a regular close; excluded sessions remain in the daily output with an explicit quality flag.

| Split | Target dates | Purpose |
|---|---|---|
| Burn-in | 2010-01-04 to 2010-02-03 | Supplies the 22-trading-day HAR feature; not evaluated. |
| Train | 2010-02-01 to 2021-12-31 | Initial estimation history. |
| Validation | 2022-01-03 to 2023-12-29 | Selects the model and estimation-window length. |
| Final OOS | 2024-01-02 to 2026-09-18 | Protected final evaluation; not used by the preliminary scripts. |

The overlap between the stated burn-in and Train calendar labels is resolved operationally: the first HAR-RV forecast is emitted only after 22 prior eligible trading sessions are available. Splits are assigned by `target_date`, not by the date on which the forecast is made. Thus the forecast for a validation target may use the immediately preceding 2021 observation, but never the validation target's data.

The baseline set is GARCH(1,1)-t, EGARCH(1,1)-t, GJR-GARCH(1,1)-t, and log-HAR-RV with daily, five-day, and 22-day RV features. The GARCH-family models use the same-day intraday session return from the one-minute file, rather than an overnight-inclusive close-to-close return. This improves target alignment but does not make the conditional-variance forecast identical to realized variance.

## Reproducible run order

Run these commands from the repository root after installing `extensions/requirements.txt`:

```bash
python extensions/scripts/build_1min_daily_rv.py
python extensions/scripts/build_features_and_splits.py
python extensions/scripts/run_baseline_models.py --window 756 --refit-every 5
python extensions/scripts/evaluate_forecasts.py
```

The initial `756`-day rolling window and five-session parameter-refit cadence are provisional. Validation should compare `504`, `756`, `1008`, and `expanding` windows using QLIKE as the primary criterion before one specification is locked. The final-OOS option in `run_baseline_models.py` intentionally raises an error until that choice is documented.

Every script validates that feature/model information ends no later than the forecast-origin date and that `forecast_origin_date < target_date`. Metrics include QLIKE, MAE, RMSE, and pairwise QLIKE Diebold–Mariano tests with a five-lag Bartlett HAC estimator.
