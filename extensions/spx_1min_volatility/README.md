# SPX One-Minute Volatility Extension

This extension will rebuild SPX realized variance from one-minute close-to-close log returns and evaluate volatility forecasts with point-in-time data only. It is separate from the repository's legacy trading notebooks and does not make a trading recommendation.

## Structure

- `code/`: exploratory and reporting notebooks.
- `scripts/`: reproducible data-building and model-running scripts.
- `data/raw/SPX_1min.csv`: versioned one-minute SPX OHLC source data.
- `data/processed/`: derived one-minute returns, session-quality diagnostics, and realized-variance datasets.
- `requirements.txt`: dependencies specific to this extension.

## Raw-data contract

`SPX_1min.csv` contains `DateTime`, `Symbol`, `Open`, `High`, `Low`, `Close`, and `Volume`. It spans 2010-01-04 through 2026-09-18. The `Volume` field is zero throughout, so it must not be used for liquidity, VWAP, or volume-weighted-volatility analysis.

The reconstruction must sort timestamps, calculate log returns only within each trading day, exclude overnight returns, preserve the number of valid intraday intervals, and flag incomplete or non-standard sessions before estimating daily realized variance.
