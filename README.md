# TS-Volatility

Research repository for S&P 500 index (SPX) conditional-volatility forecasting. The repository distinguishes the canonical forecast evaluation from legacy model exploration and trading backtests. It is a historical research project, not a trading system or investment recommendation.

## Canonical workflow

`code/spx_rv_forecast_leakage_free.ipynb` is the canonical model-evaluation notebook. It compares calibrated EWMA, skew-t EGARCH(1,1), and skew-t GJR-GARCH(1,1) forecasts against same-day intraday realized variance (RV).

- Train period: 2023-01-04 through 2024-06-28, 373 sessions.
- Test period: 2024-07-01 through 2024-12-31, 128 sessions.
- Target: intraday RV, defined as the sum of squared 30-minute close-to-close log returns. It excludes overnight close-to-open variance.
- Forecast timing: for test date *t*, every forecast uses return data through *t - 1* only. The notebook asserts this condition for every test date.
- Evaluation: QLIKE is the primary loss. MSE, RMSE, MAE, and pairwise HAC-based QLIKE comparisons are supplementary.

The canonical notebook does not make a trading-strategy claim. Model ranking is valid only after the notebook is rerun with the versioned inputs and the reported diagnostic checks pass.

## Reproduce the canonical evaluation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_canonical_rv_data.py
jupyter nbconvert --to notebook --execute --inplace code/spx_rv_forecast_leakage_free.ipynb
```

The build script reconstructs the canonical processed CSVs and checks the expected sample sizes, non-overlapping train/test dates, positive prices, unique timestamps, and one-to-one test/RV coverage.

## Data lineage

| Input or output | Period | Role |
|---|---:|---|
| `data/raw/SP_Index_500.csv` | 2015-12-31 to 2025-12-31 | Source of daily SPX close levels and canonical log returns. |
| `data/raw/spx_intraday_30min_2024_h2.xlsx` | 2024 H2 | Versioned 30-minute intraday SPX prices used to construct RV. |
| `data/processed/canonical/spx_close_returns_train_2023-01-04_to_2024-06-28.csv` | 2023-01 to 2024-06 | Canonical training returns. |
| `data/processed/canonical/spx_close_returns_test_2024-07-01_to_2024-12-31.csv` | 2024 H2 | Canonical test returns. |
| `data/processed/canonical/spx_realized_variance_2024-07-01_to_2024-12-31.csv` | 2024 H2 | Same-day RV target and intraday interval count. |

The legacy filenames beginning with `SPY_` contain SPX index levels, not the SPY ETF. They are retained to avoid breaking historical notebooks; canonical inputs use `spx_` names.

## Model status

| Notebook | Status | Methodological status |
|---|---|---|
| `code/spx_rv_forecast_leakage_free.ipynb` | Canonical | Expanding one-step-ahead forecasts, train-only EWMA calibration, point-in-time assertions, and true intraday RV. |
| `code/Volatility_Model.ipynb` | Legacy | GARCH-family forecasts use an expanding window, but its EWMA update records the current test return before evaluation. It also evaluates against squared daily returns rather than intraday RV. |
| `code/Intraday_Volatility_Model.ipynb` | Legacy | Uses sequential forecast updates, but converts intraday average volatility into a variance proxy rather than constructing RV directly. |
| `E-Garch-Maria.ipynb`, `GJR Garch.ipynb`, `EWMA (Jan-December 2025).ipynb`, `EWMA Model-2026.ipynb` | Legacy trading research | Not canonical volatility-model evidence. Their strategy thresholds, contract representation, and performance comparisons require separate remediation. |
| `trading_Alex_final_4extended.ipynb` | Trading backtest under remediation | Requires executable option-contract data and non-zero hedge trading costs before performance claims can be relied on. |

## Repository structure

- `code/`: notebooks for EDA, mean models, canonical volatility evaluation, and legacy research.
- `scripts/build_canonical_rv_data.py`: deterministic canonical-data reconstruction and checks.
- `data/raw/`: source price files.
- `data/processed/canonical/`: versioned, model-ready inputs for the canonical notebook.
- `Literature_Review/`: background research and references.
- `requirements.txt`: Python dependencies for the canonical workflow.
- `extensions/`: isolated SPX one-minute realized-volatility reconstruction and forecasting project.
