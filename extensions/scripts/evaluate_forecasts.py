#!/usr/bin/env python3
"""Score variance forecasts and run pairwise Diebold-Mariano tests on QLIKE loss."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def qlike(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    forecast = np.maximum(forecast, 1e-12)
    return np.log(forecast) + realized / forecast


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int = 5) -> tuple[float, float]:
    differential = loss_a - loss_b
    n = len(differential)
    centered = differential - differential.mean()
    long_run_variance = np.dot(centered, centered) / n
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = np.dot(centered[lag:], centered[:-lag]) / n
        long_run_variance += 2 * (1 - lag / (max_lag + 1)) * covariance
    statistic = differential.mean() / np.sqrt(max(long_run_variance, 1e-18) / n)
    return float(statistic), float(2 * norm.sf(abs(statistic)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("extensions/data/processed/validation_forecasts.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("extensions/data/processed/validation_metrics.csv"))
    parser.add_argument("--dm-output", type=Path, default=Path("extensions/data/processed/validation_dm_qlike.csv"))
    args = parser.parse_args()

    forecasts = pd.read_csv(args.input, parse_dates=["forecast_origin_date", "target_date"])
    if forecasts["split"].eq("final_oos").any():
        raise AssertionError("Final OOS forecasts must not be used in this preliminary evaluation.")
    if not (forecasts["forecast_origin_date"] < forecasts["target_date"]).all():
        raise AssertionError("Look-ahead check failed.")
    forecasts["qlike"] = qlike(
        forecasts["realized_variance"].to_numpy(), forecasts["forecast_variance"].to_numpy()
    )
    forecasts["absolute_error"] = abs(forecasts["realized_variance"] - forecasts["forecast_variance"])
    forecasts["squared_error"] = (forecasts["realized_variance"] - forecasts["forecast_variance"]) ** 2
    metrics = (
        forecasts.groupby(["model", "window", "refit_every"], observed=True)
        .agg(
            n_forecasts=("target_date", "size"),
            qlike=("qlike", "mean"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
        )
        .sort_values("qlike")
        .reset_index()
    )
    loss_by_model = forecasts.pivot(index="target_date", columns="model", values="qlike").dropna()
    dm_rows = []
    for left, right in combinations(loss_by_model.columns, 2):
        statistic, p_value = dm_test(loss_by_model[left].to_numpy(), loss_by_model[right].to_numpy())
        dm_rows.append({"model_a": left, "model_b": right, "n_dates": len(loss_by_model), "dm_statistic": statistic, "p_value": p_value})
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False)
    pd.DataFrame(dm_rows).to_csv(args.dm_output, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
