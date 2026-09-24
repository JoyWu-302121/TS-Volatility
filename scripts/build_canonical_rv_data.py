"""Build the versioned 2024 H2 SPX forecasting inputs from raw repository data."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_SOURCE = PROJECT_ROOT / "data/raw/SP_Index_500.csv"
INTRADAY_SOURCE = PROJECT_ROOT / "data/raw/spx_intraday_30min_2024_h2.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/canonical"

TRAIN_START = pd.Timestamp("2023-01-04")
TRAIN_END = pd.Timestamp("2024-06-28")
TEST_START = pd.Timestamp("2024-07-01")
TEST_END = pd.Timestamp("2024-12-31")


def build_close_return_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(DAILY_SOURCE, parse_dates=["Date"])
    required = {"Date", "PX_LAST"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily source is missing columns: {sorted(missing)}")

    daily = (
        daily.loc[:, ["Date", "PX_LAST"]]
        .rename(columns={"PX_LAST": "close"})
        .dropna()
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )
    if (daily["close"] <= 0).any():
        raise ValueError("Daily source contains non-positive closes.")
    daily["log_return"] = np.log(daily["close"]).diff()

    train = daily.loc[daily["Date"].between(TRAIN_START, TRAIN_END)].copy()
    test = daily.loc[daily["Date"].between(TEST_START, TEST_END)].copy()
    for name, frame, expected_rows in (("train", train, 373), ("test", test, 128)):
        if len(frame) != expected_rows or frame["log_return"].isna().any():
            raise ValueError(f"Unexpected {name} sample: {len(frame)} rows or missing returns.")
    if not train["Date"].max() < test["Date"].min():
        raise ValueError("Train and test samples overlap or are not ordered.")
    return train, test


def build_realized_variance() -> pd.DataFrame:
    intraday = pd.read_excel(INTRADAY_SOURCE, parse_dates=["Date"])
    required = {"Date", "Close"}
    missing = required.difference(intraday.columns)
    if missing:
        raise ValueError(f"Intraday source is missing columns: {sorted(missing)}")
    intraday = intraday.loc[:, ["Date", "Close"]].dropna().sort_values("Date").copy()
    if intraday["Date"].duplicated().any() or (intraday["Close"] <= 0).any():
        raise ValueError("Intraday source has duplicate timestamps or non-positive closes.")

    intraday["trade_date"] = intraday["Date"].dt.normalize()
    intraday["intraday_log_return"] = intraday.groupby("trade_date")["Close"].transform(
        lambda prices: np.log(prices / prices.shift(1))
    )
    rv = (
        intraday.assign(squared_return=intraday["intraday_log_return"].pow(2))
        .groupby("trade_date", as_index=False)
        .agg(
            interval_returns=("intraday_log_return", "count"),
            realized_variance=("squared_return", "sum"),
        )
        .rename(columns={"trade_date": "Date"})
    )
    rv["realized_volatility"] = np.sqrt(rv["realized_variance"])
    if len(rv) != 128 or rv["realized_variance"].le(0).any():
        raise ValueError("Unexpected realized-variance sample coverage or values.")
    return rv


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = build_close_return_samples()
    rv = build_realized_variance()
    if not test["Date"].reset_index(drop=True).equals(rv["Date"].reset_index(drop=True)):
        raise ValueError("Test returns and realized variance do not share identical dates.")

    train.to_csv(OUTPUT_DIR / "spx_close_returns_train_2023-01-04_to_2024-06-28.csv", index=False)
    test.to_csv(OUTPUT_DIR / "spx_close_returns_test_2024-07-01_to_2024-12-31.csv", index=False)
    rv.to_csv(OUTPUT_DIR / "spx_realized_variance_2024-07-01_to_2024-12-31.csv", index=False)
    print(f"Wrote {len(train)} training rows, {len(test)} test rows, and {len(rv)} RV rows.")


if __name__ == "__main__":
    main()
