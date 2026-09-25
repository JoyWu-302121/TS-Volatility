#!/usr/bin/env python3
"""Write a concise, data-derived Validation-only research summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("extensions/data/processed/validation_metrics.csv"))
    parser.add_argument("--dm", type=Path, default=Path("extensions/data/processed/validation_dm_qlike.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("extensions/data/processed/spx_split_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("extensions/reports/validation_summary.md"))
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics).sort_values("qlike")
    dm = pd.read_csv(args.dm)
    manifest = pd.read_csv(args.manifest).set_index("split")
    winner = metrics.iloc[0]
    winner_pair = dm.loc[(dm["model_a"].eq(winner["model"])) | (dm["model_b"].eq(winner["model"]))].copy()
    validation = manifest.loc["validation"]
    final_oos = manifest.loc["final_oos"]
    metric_rows = "\n".join(
        f"| {row.model} | {row.qlike:.6f} | {row.mae:.8f} | {row.rmse:.8f} |"
        for row in metrics.itertuples()
    )
    dm_rows = "\n".join(
        f"| {row.model_a} vs {row.model_b} | {row.dm_statistic:.4f} | {row.p_value:.6g} |"
        for row in winner_pair.itertuples()
    )
    report = "# Validation summary\n\n"
    report += "**Status:** preliminary Validation-only comparison. It is not a Final OOS result and must not be used as a trading recommendation.\n\n"
    report += "## Configuration\n\n"
    report += f"- Target dates: {validation.first_target_date} to {validation.last_target_date} ({validation.n_forecasts} forecasts).\n"
    report += f"- Estimation: {winner.window}-trading-day rolling window, refit every {winner.refit_every} sessions.\n"
    report += "- Models: GARCH(1,1)-t, EGARCH(1,1)-t, GJR-GARCH(1,1)-t, and log-HAR-RV.\n"
    report += "- Target: next quality-approved session's intraday realized variance. QLIKE is the primary selection loss; lower is better.\n\n"
    report += "## Results\n\n| Model | QLIKE | MAE | RMSE |\n|---|---:|---:|---:|\n" + metric_rows + "\n\n"
    report += f"The lowest Validation QLIKE is **{winner.model}** ({winner.qlike:.6f}). This is a candidate, not a locked final model, because alternative windows have not yet been compared.\n\n"
    report += "## QLIKE Diebold–Mariano comparisons involving the current leader\n\n| Comparison | DM statistic | Two-sided p-value |\n|---|---:|---:|\n" + dm_rows + "\n\n"
    report += "A positive statistic means the first listed model has a higher QLIKE loss than the second.\n\n"
    report += "## Protected final holdout\n\n"
    report += f"The Final OOS target interval ({final_oos.first_target_date} to {final_oos.last_target_date}; {final_oos.n_forecasts} dates) is present only in the split manifest and has not been forecast, scored, or used to choose this configuration.\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}.")


if __name__ == "__main__":
    main()
