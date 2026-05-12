import pandas as pd
import numpy as np
import pytest

def make_sample_df(n=300):
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low':  close * 0.98,
        'Close': close,
        'Adj Close': close,
        'Volume': np.random.randint(1_000_000, 5_000_000, n)
    }, index=dates)

def test_sma_length():
    try:
        import talib
        df = make_sample_df()
        sma = talib.SMA(df['Adj Close'].values.astype(float), timeperiod=20)
        assert len(sma) == len(df)
    except ImportError:
        df = make_sample_df()
        sma = pd.Series(df['Adj Close']).rolling(window=20).mean()
        assert len(sma) == len(df)

def test_rsi_range():
    try:
        import talib
        df = make_sample_df()
        rsi = talib.RSI(df['Adj Close'].values.astype(float), timeperiod=14)
        valid = rsi[~np.isnan(rsi)]
        assert valid.min() >= 0
        assert valid.max() <= 100
    except ImportError:
        df = make_sample_df()
        delta = pd.Series(df['Adj Close']).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        valid = rsi[~np.isnan(rsi)]
        assert valid.min() >= 0
        assert valid.max() <= 100

def test_daily_return_calculation():
    df = make_sample_df()
    df['daily_return'] = df['Adj Close'].pct_change() * 100
    assert df['daily_return'].iloc[0] != df['daily_return'].iloc[0]  # first is NaN
    assert abs(df['daily_return'].iloc[1:].mean()) < 5  # reasonable range

def test_no_null_close_after_clean():
    df = make_sample_df()
    df = df.dropna(subset=['Close', 'Adj Close'])
    assert df['Close'].isnull().sum() == 0
