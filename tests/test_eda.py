import pandas as pd
import pytest

def test_headline_length_column():
    df = pd.DataFrame({'headline': ['Stock hits all-time high', 'Earnings beat expectations']})
    df['headline_length'] = df['headline'].apply(len)
    assert df['headline_length'].iloc[0] == 24

def test_no_null_headlines():
    df = pd.DataFrame({'headline': ['Valid headline', None]})
    df = df.dropna(subset=['headline'])
    assert df.shape[0] == 1

def test_domain_extraction():
    pub = 'john@reuters.com'
    domain = pub.split('@')[-1]
    assert domain == 'reuters.com'
