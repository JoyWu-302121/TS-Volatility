#!/usr/bin/env python3
"""Create one-step-ahead HAR-RV features and target-date-based sample splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_END = pd.Timestamp("2021-12-31")
VALIDATION_START = pd.Timestamp("2022-01-03")
VALIDATION_END = pd.Timestamp("2023-12-29")
FINAL_OOS_START = pd.Timestamp("2024-01-02")
FINAL_OOS_END = pd.Timestamp("2026-09-18")
MONTHLY_LOOKBACK = 22


def split_for_target(target_date: pd.Timestamp) -> str:
    if target_date <= TRAIN_END:
        return "train"
    if VALIDATION_START <= target_date <= VALIDATION_END:
        return "validation"
    if FINAL_OOS_START <= target_date <= FINAL_OOS_END:
        return "final_oos"
    return "excluded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("extensions/data/processed/spx_1min_daily_rv.csv"))
    parser.add_argument("--dataset-output", type=Path, default=Path("extensions/data/processed/spx_daily_model_dataset.csv"))
    parser.add_argument("--manifest-output", type=Path, default=Path("extensions/data/processed/spx_split_manifest.csv"))
    args = parser.parse_args()

    daily = pd.read_csv(args.input, parse_dates=["date"])
    eligible = daily["is_model_eligible"].astype(str).str.lower().eq("true")
    daily = daily.loc[eligible].copy().sort_values("date")
    daily["rv_d"] = daily["rv"]
    daily["rv_w"] = daily["rv"].rolling(5, min_periods=5).mean()
    daily["rv_m"] = daily["rv"].rolling(MONTHLY_LOOKBACK, min_periods=MONTHLY_LOOKBACK).mean()
    daily["target_date"] = daily["date"].shift(-1)
    daily["target_rv"] = daily["rv"].shift(-1)
    daily["split"] = daily["target_date"].map(split_for_target)
    dataset = daily.dropna(subset=["rv_w", "rv_m", "target_date", "target_rv"]).copy()
    dataset["log_rv_d"] = np.log(dataset["rv_d"])
    dataset["log_rv_w"] = np.log(dataset["rv_w"])
    dataset["log_rv_m"] = np.log(dataset["rv_m"])
    dataset = dataset.rename(columns={"date": "forecast_origin_date"})
    keep = [
        "forecast_origin_date", "target_date", "split", "target_rv", "session_return",
        "rv_d", "rv_w", "rv_m", "log_rv_d", "log_rv_w", "log_rv_m",
        "n_intraday_returns", "quality_flag",
    ]
    dataset = dataset[keep]
    if not (dataset["forecast_origin_date"] < dataset["target_date"]).all():
        raise ValueError("Forecast origin must precede target date.")
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.dataset_output, index=False, date_format="%Y-%m-%d")
    manifest = (
        dataset.groupby("split", observed=True)
        .agg(
            n_forecasts=("target_date", "size"),
            first_target_date=("target_date", "min"),
            last_target_date=("target_date", "max"),
            first_origin_date=("forecast_origin_date", "min"),
            last_origin_date=("forecast_origin_date", "max"),
        )
        .reset_index()
    )
    manifest.to_csv(args.manifest_output, index=False, date_format="%Y-%m-%d")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
