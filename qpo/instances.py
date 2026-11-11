"""Problem instances drawn from real price history, split by time.

"train" instances use windows that end on or before 2023-12-31; "test"
instances use windows that start on or after 2024-01-01. Anything that is
fitted (the PPO policy, default angles) only ever sees "train".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import annualised_stats
from .problem import PortfolioProblem

SPLIT_DATE = pd.Timestamp("2024-01-01")
WINDOW = 252          # one trading year per instance


def _start_range(prices: pd.DataFrame, split: str, window: int):
    first_test = int(np.searchsorted(prices.index, SPLIT_DATE))
    if split == "train":
        return 0, first_test - window
    if split == "test":
        return first_test, len(prices) - window
    raise ValueError("split must be 'train' or 'test'")


def sample_problem(prices: pd.DataFrame, rng: np.random.Generator, n: int, budget: int,
                   risk_aversion: float, split: str, window: int = WINDOW) -> PortfolioProblem:
    """One instance: n random tickers (in random qubit order) over one random window."""
    lo, hi = _start_range(prices, split, window)
    start = int(rng.integers(lo, hi + 1))
    tickers = tuple(rng.choice(list(prices.columns), size=n, replace=False))
    mu, sigma, names = annualised_stats(prices.iloc[start:start + window], tickers)
    return PortfolioProblem(mu=mu, sigma=sigma, budget=budget, risk_aversion=risk_aversion,
                            tickers=names)


def make_pool(prices: pd.DataFrame, split: str, count: int, n: int, budget: int,
              risk_aversion: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [sample_problem(prices, rng, n, budget, risk_aversion, split) for _ in range(count)]
