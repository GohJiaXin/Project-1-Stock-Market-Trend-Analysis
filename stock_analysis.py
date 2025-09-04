"""
stock_analysis.py
-----------------
Core functions for Stock Market Trend Analysis.

Features:
- Simple Moving Average (SMA)
- Daily returns
- Consecutive upward and downward runs
- Best Time to Buy and Sell Stock II (max profit with multiple transactions)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    """
    Compute Simple Moving Average (SMA) for a given window size.
    """
    return close.rolling(window=window, min_periods=window).mean()


def compute_daily_returns(close: pd.Series) -> pd.Series:
    """
    Compute simple daily returns:
    r_t = (P_t - P_{t-1}) / P_{t-1}
    """
    return close.pct_change()


@dataclass
class Run:
    direction: str           # "up" or "down"
    start_idx: int
    end_idx: int
    length: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def compute_runs(close: pd.Series) -> Tuple[List[Run], Dict[str, int], Dict[str, Optional[Run]]]:
    """
    Find consecutive upward and downward runs.
    - Up run: close[t] > close[t-1]
    - Down run: close[t] < close[t-1]
    Equal closes break the run.

    Returns:
        runs: list of Run objects
        counts: number of runs and total days
        longest: longest up and down run
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        close = close.copy()
        close.index = pd.to_datetime(close.index)

    diff = close.diff()
    runs: List[Run] = []
    n = len(close)

    if n <= 1:
        return runs, {"up_runs": 0, "down_runs": 0, "up_days": 0, "down_days": 0}, {"up": None, "down": None}

    current_dir: Optional[str] = None
    run_start = 1

    for i in range(1, n):
        if pd.isna(diff.iloc[i]):
            continue

        step_dir = "up" if diff.iloc[i] > 0 else ("down" if diff.iloc[i] < 0 else None)

        if current_dir is None:
            if step_dir is not None:
                current_dir = step_dir
                run_start = i
        else:
            if step_dir != current_dir:
                length = i - run_start
                if length > 0:
                    runs.append(
                        Run(
                            direction=current_dir,
                            start_idx=run_start - 1,
                            end_idx=i - 1,
                            length=length,
                            start_date=close.index[run_start - 1],
                            end_date=close.index[i - 1],
                        )
                    )
                current_dir = step_dir
                run_start = i if step_dir is not None else None

    if current_dir is not None and run_start is not None:
        length = n - run_start
        if length > 0:
            runs.append(
                Run(
                    direction=current_dir,
                    start_idx=run_start - 1,
                    end_idx=n - 1,
                    length=length,
                    start_date=close.index[run_start - 1],
                    end_date=close.index[n - 1],
                )
            )

    up_runs = [r for r in runs if r.direction == "up"]
    down_runs = [r for r in runs if r.direction == "down"]

    counts = {
        "up_runs": len(up_runs),
        "down_runs": len(down_runs),
        "up_days": sum(r.length for r in up_runs),
        "down_days": sum(r.length for r in down_runs),
    }
    longest = {
        "up": max(up_runs, key=lambda r: r.length) if up_runs else None,
        "down": max(down_runs, key=lambda r: r.length) if down_runs else None,
    }
    return runs, counts, longest


def max_profit_stock_ii(close: pd.Series) -> Tuple[float, list]:
    """
    Best Time to Buy and Sell Stock II (multiple transactions allowed).
    Profit = sum of all positive day-to-day increases.

    Returns:
        profit (float), trades (list of (buy_idx, sell_idx))
    """
    if len(close) < 2:
        return 0.0, []

    prices = close.values
    profit = 0.0
    trades = []
    i = 0
    n = len(prices)

    while i < n - 1:
        while i < n - 1 and prices[i + 1] <= prices[i]:
            i += 1
        buy = i
        while i < n - 1 and prices[i + 1] >= prices[i]:
            i += 1
        sell = i
        if sell > buy:
            profit += prices[sell] - prices[buy]
            trades.append((buy, sell))

    return float(profit), trades
