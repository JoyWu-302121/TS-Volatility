# TS-Volatility
Time-Series Model for Volatility Simulation 

This project develops a time-series modeling framework to analyze and forecast equity market volatility using S&P 500 (SPX/SPY) data. The goal is to understand return dynamics and generate volatility forecasts that can be applied to trading strategies and risk management.

The workflow begins with data collection and cleaning, followed by exploratory data analysis (EDA) to examine return distributions, autocorrelation, and volatility clustering. We then estimate mean models using ARIMA to capture the underlying return process. After filtering the mean, we model conditional volatility using GARCH-type models to capture time-varying variance.

The models are evaluated using statistical diagnostics such as AIC/BIC, residual analysis, and Ljung–Box tests. Finally, the volatility forecasts are compared to market-implied measures and used to motivate potential trading strategies, such as volatility-based positioning.

## Project Structure

- `data/`  
 Contains the raw and processed datasets used throughout the project, including SPX/SPY price data and any cleaned return series used for modeling.

- `Literature_Review/`  
  Contains background research and references on time-series modeling, volatility, and GARCH-type models that inform the methodology used in this project.

- `README.md`  
  Provides an overview of the project, methodology, and repository structure.
  
- `code/`
  Includes supporting scripts and functions used for data processing, model estimation, and analysis.
## Data
  Contains the raw and processed datasets used throughout the project, including original price data, cleaned return series, training and test datasets, intraday volatility data for reference, and options data containing implied volatility.

# `data/raw/`
Contains the original source files before cleaning or transformation.

- `SP_Index_500.csv`  
  Raw historical S&P 500 price data used as the starting point for the time-series analysis.

- `Intraday_vol.xlsx`  
  Raw intraday volatility data used for informational purposes. This file provides additional context on intraday market volatility, but it is not the main input for the core ARIMA and GARCH modeling workflow.

# `data/processed/`
Contains cleaned and model-ready datasets used throughout the project.

- `SPY_Close.csv`  
  Cleaned daily SPY closing price data used to construct returns and support the time-series modeling process.

- `SPY_log_return.csv`  
  Daily log returns computed from SPY closing prices. This is the main return series used for mean and volatility modeling.

- `SPY_Close_modeling_2.5y.csv`  
  Training dataset covering the 2.5-year modeling sample. This file is used to fit the time-series models.

- `SPY_Close_forecast_0.5y.csv`  
  Test dataset covering the 0.5-year out-of-sample period. This file is used to evaluate forecast performance.

- `SPX_Intraday_Vol.csv`  
  Processed intraday volatility dataset included for informational purposes. It provides additional insight into volatility behavior within the trading day, rather than serving as a primary input to the core models.

- `options_data_2025.xlsx`  
  Processed options dataset containing implied volatility data. This file is used to compare model-based volatility forecasts with market-implied volatility.
- `code/`  
  Includes supporting scripts and functions used for data processing, model estimation, and analysis.

- `Literature_Review/`  
  Contains background research and references on time-series modeling, volatility, and GARCH-type models that inform the methodology used in this project.

- `README.md`  
  Provides an overview of the project, methodology, and repository structure.

## Code  
  Includes supporting scripts and functions used for data processing, model estimation, and analysis.
  
- `EDA_SPY.ipynb`  
  Performs exploratory data analysis on SPY price and return data. This includes computing log returns, visualizing return distributions, and analyzing autocorrelation and volatility clustering to motivate time-series modeling.

- Mean_Model.ipynb`  
  Focuses on a shorter, more recent time period to better capture current market conditions. Used for model refinement, forecasting, and comparison with the long-horizon model.

- `Volatility_Model.ipynb`  
  Models time-varying volatility using GARCH-type models on the residuals from the mean model. Generates volatility forecasts and evaluates model performance, with applications to risk measurement and volatility-based trading strategies.

  - `intra_vol_process.ipynb`  
  Processes high-frequency intraday data to construct intraday volatility measures used for analysis and modeling.

- `Intraday_Volatility_Model.ipynb`  
  Implements a model to analyze and forecast intraday volatility dynamics using the processed high-frequency data.
