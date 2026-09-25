#!/usr/bin/env python3
"""Create deterministic figures for the protected Validation-only model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["HAR-RV", "GJR-GARCH", "GARCH", "EGARCH"]
COLORS = {"HAR-RV": "#1b9e77", "GJR-GARCH": "#d95f02", "GARCH": "#7570b3", "EGARCH": "#e7298a"}


def annualized_volatility(variance: pd.Series) -> pd.Series:
    return np.sqrt(variance.clip(lower=0) * 252.0) * 100.0


def qlike(realized: pd.Series, forecast: pd.Series) -> pd.Series:
    forecast = forecast.clip(lower=1e-12)
    return np.log(forecast) + realized / forecast


def save_forecast_path(forecasts: pd.DataFrame, output: Path) -> None:
    wide = forecasts.pivot(index="target_date", columns="model", values="forecast_variance").reindex(columns=MODEL_ORDER)
    actual = forecasts.groupby("target_date")["realized_variance"].first()
    plot_data = pd.DataFrame({"Realized RV": annualized_volatility(actual)})
    for model in MODEL_ORDER:
        plot_data[model] = annualized_volatility(wide[model])
    smooth = plot_data.rolling(21, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(smooth.index, smooth["Realized RV"], color="#222222", lw=2.1, label="Realized RV (21D mean)")
    for model in MODEL_ORDER:
        ax.plot(smooth.index, smooth[model], color=COLORS[model], lw=1.3, label=f"{model} (21D mean)")
    ax.set(title="Validation: realized vs forecast annualized volatility", xlabel="Target date", ylabel="Annualized volatility (%)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "validation_rv_forecast_path.png", dpi=180)
    plt.close(fig)


def save_metric_comparison(metrics: pd.DataFrame, output: Path) -> None:
    metrics = metrics.set_index("model").reindex(MODEL_ORDER).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    qlike_axis = axes[0]
    y = np.arange(len(metrics))
    qlike_axis.hlines(y, metrics["qlike"].min(), metrics["qlike"], color="#bdbdbd", lw=1.4)
    qlike_axis.scatter(metrics["qlike"], y, s=70, color=[COLORS[m] for m in metrics["model"]], zorder=3)
    for x, y_value in zip(metrics["qlike"], y):
        qlike_axis.annotate(f"{x:.4f}", (x, y_value), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    qlike_axis.set(
        title="Mean QLIKE (lower is better)",
        yticks=y,
        yticklabels=metrics["model"],
        xlim=(metrics["qlike"].min() - 0.006, metrics["qlike"].max() + 0.018),
    )
    qlike_axis.invert_yaxis()
    qlike_axis.grid(axis="x", alpha=0.22)
    for ax, column, label in zip(axes[1:], ["mae", "rmse"], ["MAE", "RMSE"]):
        bars = ax.bar(metrics["model"], metrics[column], color=[COLORS[m] for m in metrics["model"]])
        ax.bar_label(bars, labels=[f"{value:.4g}" for value in metrics[column]], padding=3, fontsize=9)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Validation loss comparison: 756-day rolling, 5-day refit", y=1.03)
    fig.tight_layout()
    fig.savefig(output / "validation_qlike_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_cumulative_qlike(forecasts: pd.DataFrame, output: Path) -> None:
    forecast_loss = forecasts.copy()
    forecast_loss["qlike"] = qlike(forecast_loss["realized_variance"], forecast_loss["forecast_variance"])
    wide = forecast_loss.pivot(index="target_date", columns="model", values="qlike").reindex(columns=MODEL_ORDER)
    baseline = wide["HAR-RV"]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axhline(0, color="#222222", lw=0.8)
    for model in MODEL_ORDER[1:]:
        ax.plot(wide.index, (wide[model] - baseline).cumsum(), color=COLORS[model], lw=1.7, label=f"{model} − HAR-RV")
    ax.set(title="Cumulative QLIKE differential vs HAR-RV", xlabel="Target date", ylabel="Cumulative loss difference")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "validation_cumulative_qlike_difference.png", dpi=180)
    plt.close(fig)


def save_calibration(forecasts: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    max_vol = annualized_volatility(pd.concat([forecasts["realized_variance"], forecasts["forecast_variance"]])).quantile(0.995)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        subset = forecasts.loc[forecasts["model"].eq(model)]
        x = annualized_volatility(subset["forecast_variance"])
        y = annualized_volatility(subset["realized_variance"])
        ax.scatter(x, y, s=10, alpha=0.35, color=COLORS[model], edgecolor="none")
        ax.plot([0, max_vol], [0, max_vol], color="#222222", lw=1)
        ax.set(title=model, xlim=(0, max_vol), ylim=(0, max_vol))
        ax.grid(alpha=0.2)
    fig.supxlabel("Forecast annualized volatility (%)")
    fig.supylabel("Realized annualized volatility (%)")
    fig.suptitle("Validation calibration", y=0.93)
    fig.tight_layout()
    fig.savefig(output / "validation_actual_vs_forecast.png", dpi=180)
    plt.close(fig)


def save_session_quality(daily: pd.DataFrame, output: Path) -> None:
    quality = daily.groupby("quality_flag", observed=True).size().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(quality.index, quality.values, color=["#1b9e77" if name == "eligible_regular_session" else "#d95f02" for name in quality.index])
    axes[0].set(title="Session quality classification", ylabel="Number of sessions")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.22)
    axes[1].hist(daily["n_intraday_returns"], bins=35, color="#7570b3", alpha=0.85)
    axes[1].axvline(300, color="#d95f02", linestyle="--", label="Eligibility threshold")
    axes[1].set(title="Intraday-return count per session", xlabel="Number of returns", ylabel="Sessions")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output / "session_quality_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, default=Path("extensions/data/processed/validation_forecasts.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("extensions/data/processed/validation_metrics.csv"))
    parser.add_argument("--daily-rv", type=Path, default=Path("extensions/data/processed/spx_1min_daily_rv.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("extensions/figures"))
    args = parser.parse_args()

    forecasts = pd.read_csv(args.forecasts, parse_dates=["forecast_origin_date", "target_date"])
    if not forecasts["split"].eq("validation").all():
        raise AssertionError("Only Validation forecasts may be visualized in this report.")
    metrics = pd.read_csv(args.metrics)
    daily = pd.read_csv(args.daily_rv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_forecast_path(forecasts, args.output_dir)
    save_metric_comparison(metrics, args.output_dir)
    save_cumulative_qlike(forecasts, args.output_dir)
    save_calibration(forecasts, args.output_dir)
    save_session_quality(daily, args.output_dir)
    print(f"Wrote 5 Validation-only figures to {args.output_dir}.")


if __name__ == "__main__":
    main()
