#!/usr/bin/env python3
"""Build daily, session-only realized variance from SPX one-minute bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REGULAR_OPEN = "09:30:00"
REGULAR_CLOSE = "16:00:00"
MIN_MODEL_RETURNS = 300


def summarize_session(session: pd.DataFrame) -> dict[str, object]:
    session = session.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    prices = session["close"].to_numpy(dtype=float)
    timestamps = session["timestamp"]
    log_returns = np.diff(np.log(prices))
    n_returns = len(log_returns)
    first_time = timestamps.iloc[0]
    last_time = timestamps.iloc[-1]
    is_regular_close = last_time.time() >= pd.Timestamp("15:59:00").time()
    is_model_eligible = bool(n_returns >= MIN_MODEL_RETURNS and is_regular_close)
    if n_returns < MIN_MODEL_RETURNS:
        quality_flag = "short_or_incomplete"
    elif not is_regular_close:
        quality_flag = "missing_regular_close"
    else:
        quality_flag = "eligible_regular_session"
    return {
        "date": first_time.normalize(),
        "first_timestamp": first_time,
        "last_timestamp": last_time,
        "n_price_observations": len(prices),
        "n_intraday_returns": n_returns,
        "rv": float(np.square(log_returns).sum()),
        "session_return": float(np.log(prices[-1] / prices[0])),
        "is_regular_close": is_regular_close,
        "is_model_eligible": is_model_eligible,
        "quality_flag": quality_flag,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("extensions/data/raw/SPX_1min.csv"))
    parser.add_argument("--output", type=Path, default=Path("extensions/data/processed/spx_1min_daily_rv.csv"))
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    sessions: list[pd.DataFrame] = []
    for chunk in pd.read_csv(args.input, usecols=["DateTime", "Close"], chunksize=args.chunksize):
        chunk["timestamp"] = pd.to_datetime(chunk["DateTime"], errors="coerce")
        chunk["close"] = pd.to_numeric(chunk["Close"], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp", "close"])
        chunk = chunk.loc[chunk["close"] > 0, ["timestamp", "close"]]
        clock = chunk["timestamp"].dt.strftime("%H:%M:%S")
        sessions.append(chunk.loc[(clock >= REGULAR_OPEN) & (clock <= REGULAR_CLOSE)])

    bars = pd.concat(sessions, ignore_index=True)
    bars["date"] = bars["timestamp"].dt.normalize()
    daily = pd.DataFrame(
        [summarize_session(day) for _, day in bars.groupby("date", sort=True)]
    ).sort_values("date")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(
        f"Wrote {len(daily):,} daily sessions to {args.output}; "
        f"eligible={int(daily['is_model_eligible'].sum()):,}, "
        f"flagged={int((~daily['is_model_eligible']).sum()):,}."
    )


if __name__ == "__main__":
    main()
