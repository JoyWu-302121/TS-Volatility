# TS-Volatility
Time-Series Model for Volatility Simulation 

This project develops a time-series modeling framework to analyze and forecast equity market volatility using S&P 500 (SPX/SPY) data. The goal is to understand return dynamics and generate volatility forecasts that can be applied to trading strategies and risk management.

The workflow begins with data collection and cleaning, followed by exploratory data analysis (EDA) to examine return distributions, autocorrelation, and volatility clustering. We then estimate mean models using ARIMA to capture the underlying return process. After filtering the mean, we model conditional volatility using GARCH-type models to capture time-varying variance.

The models are evaluated using statistical diagnostics such as AIC/BIC, residual analysis, and Ljung–Box tests. Finally, the volatility forecasts are compared to market-implied measures and used to motivate potential trading strategies, such as volatility-based positioning.

## Project Structure

- `data/`  
  Contains the raw and processed datasets used throughout the project, including SPX/SPY price data and any cleaned return series used for modeling.

- `code/`  
  Includes supporting scripts and functions used for data processing, model estimation, and analysis.

- `Literature_Review/`  
  Contains background research and references on time-series modeling, volatility, and GARCH-type models that inform the methodology used in this project.

- `10yr_Model.ipynb`  
  Implements time-series modeling on a longer historical dataset (~10 years). Includes mean model estimation (ARIMA), diagnostic testing, and volatility modeling to capture long-term dynamics.

- `Final_Short_Model.ipynb`  
  Focuses on a shorter, more recent time period to better capture current market conditions. Used for model refinement, forecasting, and comparison with the long-horizon model.

- `README.md`  
  Provides an overview of the project, methodology, and repository structure.

- `EDA_SPY.ipynb`  
  Performs exploratory data analysis on SPY price and return data. This includes computing log returns, visualizing return distributions, and analyzing autocorrelation and volatility clustering to motivate time-series modeling.

- `10yr_Model.ipynb`  
  Implements time-series modeling on a longer historical dataset (~10 years). Includes mean model estimation (ARIMA), diagnostic testing, and volatility modeling to capture long-term dynamics.

- `Final_Short_Model.ipynb`  
  Focuses on a shorter, more recent time period to better capture current market conditions. Used for model refinement, forecasting, and comparison with the long-horizon model.
- `Mean_Model.ipynb`  
  Implements mean modeling of returns using ARIMA models. Includes model selection using AIC/BIC, parameter estimation, and diagnostic testing (e.g., residual analysis and Ljung–Box tests) to confirm that residuals behave like white noise.

- `Volatility_Model.ipynb`  
  Models time-varying volatility using GARCH-type models on the residuals from the mean model. Generates volatility forecasts and evaluates model performance, with applications to risk measurement and volatility-based trading strategies.
