"""Utilities for building StrategyData inputs from Yahoo Finance data.

The loader fetches daily price/volume history via yfinance and constructs
cross-sectional features on the same daily cadence so downstream components
operate with trading-day windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "yfinance is required for the YFinance data loader. "
        "Install it with `pip install yfinance`."
    ) from exc


DEFAULT_TICKERS: Tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "HD",
    "MA",
    "PFE",
    "KO",
    "PEP",
    "NFLX",
    "DIS",
    "BAC",
    "CSCO",
)

MARKET_TICKER = "SPY"

SECTOR_MAP: Mapping[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication",
    "AMZN": "ConsumerDiscretionary",
    "META": "Communication",
    "NVDA": "Technology",
    "TSLA": "ConsumerDiscretionary",
    "JPM": "Financials",
    "JNJ": "Healthcare",
    "V": "Financials",
    "PG": "ConsumerStaples",
    "HD": "ConsumerDiscretionary",
    "MA": "Financials",
    "PFE": "Healthcare",
    "KO": "ConsumerStaples",
    "PEP": "ConsumerStaples",
    "NFLX": "Communication",
    "DIS": "Communication",
    "BAC": "Financials",
    "CSCO": "Technology",
}


@dataclass
class YFinancePanels:
    returns_panel: pd.DataFrame
    risk_factors_panel: Dict[str, pd.DataFrame]
    alpha_factors_panel: Dict[str, pd.DataFrame]


def load_yfinance_panels(
    tickers: Sequence[str] | None = None,
    start: str = "2014-01-01",
    end: str = "2024-12-31",
) -> YFinancePanels:
    """Download data and build panel structures.

    Parameters
    ----------
    tickers:
        Universe of tickers to download. Defaults to a curated list of
        large-cap US equities.
    start, end:
        Date bounds (inclusive) for the download.
    """

    tickers = tuple(tickers or DEFAULT_TICKERS)
    missing = set(tickers) - set(SECTOR_MAP)
    if missing:
        raise ValueError(
            "All tickers must exist in SECTOR_MAP for sector exposures. "
            f"Missing: {sorted(missing)}"
        )

    download_symbols = sorted(set(tickers) | {MARKET_TICKER})
    data = yf.download(
        download_symbols,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError("No data returned from yfinance. Check ticker universe and date range.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs("Adj Close", axis=1, level=-1).dropna(how="all")
        volumes = data.xs("Volume", axis=1, level=-1).dropna(how="all")
    else:
        prices = data["Adj Close"].dropna(how="all")
        volumes = data["Volume"].dropna(how="all")

    ordered_symbols = list(tickers) + [MARKET_TICKER]
    prices = prices.loc[:, ordered_symbols].ffill()
    volumes = volumes.loc[:, ordered_symbols].fillna(0.0)

    market_prices = prices[MARKET_TICKER]
    prices = prices[list(tickers)]
    volumes = volumes[list(tickers)]

    daily_returns = prices.pct_change()
    market_returns = market_prices.pct_change()
    forward_returns = daily_returns.shift(-1)

    beta_lookback = 60
    beta_cov = daily_returns.rolling(beta_lookback).cov(market_returns)
    beta_var = market_returns.rolling(beta_lookback).var()
    beta = beta_cov.div(beta_var, axis=0)

    vol_3m = daily_returns.rolling(63).std()
    vol_6m = daily_returns.rolling(126).std()
    mom_3m = prices.pct_change(63)
    mom_6m = prices.pct_change(126)
    mom_12m = prices.pct_change(252)
    price_to_sma_200 = prices / prices.rolling(200).mean()

    avg_vol_21 = volumes.rolling(21).mean()
    avg_vol_63 = volumes.rolling(63).mean()
    volume_trend = avg_vol_21 / (avg_vol_63 + 1e-9)

    features = {
        "beta": beta,
        "vol_3m": vol_3m,
        "vol_6m": vol_6m,
        "mom_3m": mom_3m,
        "mom_6m": mom_6m,
        "mom_12m": mom_12m,
        "price_to_sma_200": price_to_sma_200,
        "volume_trend": volume_trend,
    }

    daily_index = forward_returns.index

    # Sector exposures as static one-hot encodings
    sectors = sorted({sector for sector in SECTOR_MAP.values() if sector})
    sector_frames: Dict[str, pd.DataFrame] = {}
    for sector in sectors:
        data_arr = np.zeros((len(daily_index), len(tickers)))
        for idx, ticker in enumerate(tickers):
            if SECTOR_MAP[ticker] == sector:
                data_arr[:, idx] = 1.0
        sector_frames[f"Sector_{sector}"] = pd.DataFrame(data_arr, index=daily_index, columns=tickers)

    risk_factors: Dict[str, pd.DataFrame] = {"MarketBeta": features["beta"]}
    risk_factors.update(sector_frames)

    alpha_factors: Dict[str, pd.DataFrame] = {
        "Alpha_Mom3M": features["mom_3m"],
        "Alpha_Mom6M": features["mom_6m"],
        "Alpha_Mom12M": features["mom_12m"],
        "Alpha_Vol3M": features["vol_3m"],
        "Alpha_Vol6M": features["vol_6m"],
        "Alpha_PriceToSMA200": features["price_to_sma_200"],
        "Alpha_VolumeTrend": features["volume_trend"],
    }

    returns_panel = forward_returns

    returns_panel, risk_factors, alpha_factors = _clean_cross_sections(
        returns_panel,
        risk_factors,
        alpha_factors,
    )

    return YFinancePanels(
        returns_panel=returns_panel,
        risk_factors_panel=risk_factors,
        alpha_factors_panel=alpha_factors,
    )


def _clean_cross_sections(
    returns_panel: pd.DataFrame,
    risk_factors: Dict[str, pd.DataFrame],
    alpha_factors: Dict[str, pd.DataFrame],
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Apply winsorization and z-scoring cross-sectionally."""

    def process_df(df: pd.DataFrame, standardize: bool = True) -> pd.DataFrame:
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)

        def transform(row: pd.Series) -> pd.Series:
            valid = row.dropna()
            if valid.empty:
                return row  # Keep NaN when all values are NaN
            lower = valid.quantile(winsor_lower)
            upper = valid.quantile(winsor_upper)
            clipped = row.clip(lower, upper)
            if not standardize:
                return clipped  # Keep NaN for non-standardized factors
            mean = clipped.mean()
            std = clipped.std(ddof=0)
            if std < 1e-8:
                return clipped - mean  # Keep NaN
            return (clipped - mean) / std  # Keep NaN

        return df.apply(transform, axis=1)

    cleaned_returns = returns_panel.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cleaned_risk: Dict[str, pd.DataFrame] = {}
    for name, df in risk_factors.items():
        # Sector dummies and MarketBeta should not be standardized
        # MarketBeta needs to preserve its absolute value (relative to SPY)
        standardize = not (name.startswith("Sector_") or name == "MarketBeta")
        cleaned_risk[name] = process_df(df, standardize=standardize)

    cleaned_alpha = {name: process_df(df, standardize=True) for name, df in alpha_factors.items()}

    common_index = cleaned_returns.index
    for frames in (cleaned_risk, cleaned_alpha):
        for key, df in frames.items():
            frames[key] = df.reindex(common_index).fillna(0.0)

    cleaned_returns = cleaned_returns.reindex(common_index).fillna(0.0)

    # Drop the final row with incomplete forward returns (NaN after shift)
    cleaned_returns = cleaned_returns.iloc[:-1]
    for frames in (cleaned_risk, cleaned_alpha):
        for key, df in frames.items():
            frames[key] = df.iloc[:-1]

    return cleaned_returns, cleaned_risk, cleaned_alpha


__all__ = ["DEFAULT_TICKERS", "load_yfinance_panels", "YFinancePanels"]
