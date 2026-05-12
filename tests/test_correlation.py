import pandas as pd
import numpy as np
import pytest
from scipy import stats

def test_daily_return_formula():
    prices = pd.Series([100, 102, 101, 105])
    returns = prices.pct_change() * 100
    assert abs(returns.iloc[1] - 2.0) < 0.001
    assert pd.isna(returns.iloc[0])

def test_pearson_correlation_range():
    np.random.seed(0)
    x = np.random.randn(100)
    y = np.random.randn(100)
    r, p = stats.pearsonr(x, y)
    assert -1.0 <= r <= 1.0
    assert 0.0 <= p <= 1.0

def test_sentiment_label_classification():
    def classify(score):
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        return 'Neutral'
    assert classify(0.5)   == 'Positive'
    assert classify(-0.5)  == 'Negative'
    assert classify(0.0)   == 'Neutral'
    assert classify(0.05)  == 'Positive'
    assert classify(-0.05) == 'Negative'

def test_merge_drops_unmatched_rows():
    news = pd.DataFrame({
        'stock': ['AAPL','AAPL','GOOG'],
        'trading_date': pd.to_datetime(['2022-01-03','2022-01-04','2022-01-03']),
        'avg_vader': [0.1, -0.2, 0.3]
    })
    ret = pd.DataFrame({
        'stock': ['AAPL','AAPL'],
        'trading_date': pd.to_datetime(['2022-01-03','2022-01-04']),
        'daily_return': [1.2, -0.5]
    })
    merged = pd.merge(news, ret, on=['stock','trading_date'], how='inner')
    assert merged.shape[0] == 2  # GOOG row dropped
    assert 'daily_return' in merged.columns
