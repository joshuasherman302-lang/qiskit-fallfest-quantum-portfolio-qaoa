"""Market data: load cached daily prices and turn them into (mu, sigma).

Real prices are cached in ``data/prices.csv`` so every run is reproducible
offline. ``fetch_prices`` refreshes the cache from Yahoo Finance.
``synthetic_prices`` is a clearly-labelled fallback (one-factor model) for
when no network or cache is available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_CSV = DATA_DIR / "prices.csv"

# 16 large caps across sectors: tech, banks, energy, health, staples, industry.
DEFAULT_TICKERS = (
    "AAPL", "MSFT", "NVDA", "AMZN",
    "JPM", "GS",
    "XOM", "CVX",
    "JNJ", "PFE", "UNH",
    "PG", "KO", "WMT",
    "CAT", "NEE",
)
TRADING_DAYS = 252


def fetch_prices(tickers=DEFAULT_TICKERS, start="2021-01-01", end="2025-12-31",
                 path: Path = PRICES_CSV) -> pd.DataFrame:
    """Download adjusted daily closes and cache them as CSV."""
    import yfinance as yf

    # threads=False: the parallel downloader can lock yfinance's sqlite cache.
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                      progress=False, threads=False)
    prices = raw["Close"][list(tickers)].dropna(how="any")
    if len(prices) < 250:
        raise RuntimeError(f"only {len(prices)} complete rows downloaded")
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.round(4).to_csv(path)
    return prices


def load_prices(path: Path = PRICES_CSV) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def synthetic_prices(n_assets: int = 16, n_days: int = 1250, seed: int = 0) -> pd.DataFrame:
    """One-factor synthetic market. NOT real data; used only as a fallback."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.6, 1.4, n_assets)
    drift = rng.uniform(0.02, 0.25, n_assets) / TRADING_DAYS
    idio = rng.uniform(0.10, 0.30, n_assets) / np.sqrt(TRADING_DAYS)
    market = rng.normal(0.0, 0.16 / np.sqrt(TRADING_DAYS), n_days)
    rets = drift + np.outer(market, beta) + rng.normal(0, 1, (n_days, n_assets)) * idio
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"SYN{i:02d}" for i in range(n_assets)])


def annualised_stats(prices: pd.DataFrame, tickers=None, start=None, end=None):
    """Annualised mean log-return vector and covariance matrix for a window."""
    window = prices.loc[start:end]
    if tickers is not None:
        window = window[list(tickers)]
    rets = np.log(window).diff().dropna()
    mu = rets.mean().to_numpy() * TRADING_DAYS
    sigma = rets.cov().to_numpy() * TRADING_DAYS
    return mu, sigma, tuple(window.columns)
