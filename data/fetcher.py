"""
data/fetcher.py
===============
Fetches stock data using yfinance.
"""

import yfinance as yf
import pandas as pd


def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for a given ticker.

    Args:
        ticker:   Stock symbol e.g. 'AAPL', 'TSLA'
        period:   '6mo', '1y', '2y', '5y'
        interval: '1d' (daily), '1wk' (weekly)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"📥 Fetching {ticker} — period={period}, interval={interval} ...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # Keep only OHLCV
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"✅ Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    return df
