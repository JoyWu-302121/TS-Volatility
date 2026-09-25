#!/usr/bin/env python3
"""Run leakage-free one-step-ahead GARCH-family and HAR-RV forecasts."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model


MODEL_SPECS = {
    "GARCH": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "EGARCH": {"vol": "EGARCH", "p": 1, "o": 0, "q": 1},
    "GJR-GARCH": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
}


def windowed(values: np.ndarray, window: str) -> np.ndarray:
    return values if window == "expanding" else values[-int(window) :]


def fit_garch(values: np.ndarray, spec: dict[str, object]):
    model = arch_model(
        values * 100.0,
        mean="Zero",
        vol=str(spec["vol"]),
        p=int(spec["p"]),
        o=int(spec["o"]),
        q=int(spec["q"]),
        dist="t",
        rescale=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp="off", show_warning=False)


def garch_forecast(values: np.ndarray, params: pd.Series, spec: dict[str, object]) -> float:
    model = arch_model(
        values * 100.0,
        mean="Zero",
        vol=str(spec["vol"]),
        p=int(spec["p"]),
        o=int(spec["o"]),
        q=int(spec["q"]),
        dist="t",
        rescale=False,
    )
    fixed = model.fix(params)
    variance = fixed.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
    return float(max(variance / 10_000.0, 1e-12))


def fit_har(training: pd.DataFrame) -> tuple[np.ndarray, float]:
    x = np.column_stack(
        [np.ones(len(training)), training[["log_rv_d", "log_rv_w", "log_rv_m"]].to_numpy()]
    )
    y = np.log(training["target_rv"].to_numpy())
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    smearing = float(np.mean(np.exp(y - x @ beta)))
    return beta, smearing


def har_forecast(row: pd.Series, beta: np.ndarray, smearing: float) -> float:
    x = np.array([1.0, row["log_rv_d"], row["log_rv_w"], row["log_rv_m"]])
    return float(max(np.exp(x @ beta) * smearing, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("extensions/data/processed/spx_daily_model_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("extensions/data/processed/validation_forecasts.csv"))
    parser.add_argument("--evaluation-split", choices=["validation", "final_oos"], default="validation")
    parser.add_argument("--window", choices=["expanding", "504", "756", "1008"], default="756")
    parser.add_argument("--refit-every", type=int, default=5)
    parser.add_argument("--min-train-observations", type=int, default=252)
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["forecast_origin_date", "target_date"])
    evaluation = data.loc[data["split"].eq(args.evaluation_split)].copy()
    if args.evaluation_split == "final_oos":
        raise ValueError("Final OOS is intentionally protected; select a validation specification first.")
    if evaluation.empty:
        raise ValueError(f"No rows for split={args.evaluation_split!r}.")

    cached_garch: dict[str, pd.Series] = {}
    cached_har: tuple[np.ndarray, float] | None = None
    records: list[dict[str, object]] = []
    for step, (_, row) in enumerate(evaluation.iterrows()):
        origin = row["forecast_origin_date"]
        # HAR outcomes are eligible only after their target is observed at the forecast origin.
        har_pool = data.loc[data["target_date"] <= origin].copy()
        return_pool = data.loc[data["forecast_origin_date"] <= origin, "session_return"].to_numpy()
        har_training = har_pool if args.window == "expanding" else har_pool.tail(int(args.window))
        returns = windowed(return_pool, args.window)
        if len(har_training) < args.min_train_observations or len(returns) < args.min_train_observations:
            continue
        if step % args.refit_every == 0:
            cached_garch = {
                name: fit_garch(returns, spec).params
                for name, spec in MODEL_SPECS.items()
            }
            cached_har = fit_har(har_training)
        if cached_har is None:
            raise RuntimeError("Model cache was not initialized.")
        base = {
            "forecast_origin_date": origin,
            "target_date": row["target_date"],
            "split": args.evaluation_split,
            "window": args.window,
            "refit_every": args.refit_every,
            "realized_variance": row["target_rv"],
        }
        for name, spec in MODEL_SPECS.items():
            records.append(
                base | {
                    "model": name,
                    "forecast_variance": garch_forecast(returns, cached_garch[name], spec),
                }
            )
        records.append(
            base
            | {
                "model": "HAR-RV",
                "forecast_variance": har_forecast(row, *cached_har),
            }
        )

    forecasts = pd.DataFrame(records)
    if forecasts.empty:
        raise ValueError("No forecasts produced; check sample dates and minimum training size.")
    if not (forecasts["forecast_origin_date"] < forecasts["target_date"]).all():
        raise AssertionError("Look-ahead check failed: forecast origin is not before target.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(
        f"Wrote {len(forecasts):,} forecasts for {forecasts['target_date'].nunique():,} dates "
        f"to {args.output}."
    )


if __name__ == "__main__":
    main()
