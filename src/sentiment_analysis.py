"""
sentiment_analysis.py
Module for macro-economic and sector-specific sentiment analysis for financial forecasting.

Dependencies:
- pandas
- numpy
- fredapi
- yfinance
"""

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
from datetime import datetime, timedelta

load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Helper: Normalize a pandas Series to z-score
def zscore(series):
    return (series - series.mean()) / series.std()

# Helper: Normalize a pandas Series to 0-1 scale
def minmax(series):
    return (series - series.min()) / (series.max() - series.min())

# --- Macro-Economic Sentiment Function ---
def get_macro_sentiment(start_date=None, end_date=None, use_zscore=True):
    """
    Fetch and combine macro-economic sentiment indicators into a composite score.
    Returns a DataFrame indexed by date with MacroSentimentScore.
    """
    fred = Fred(api_key=FRED_API_KEY)
    if end_date is None:
        end_date = datetime.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365*2)  # Default: last 2 years
    # Fetch FRED indicators
    try:
        umcsent = fred.get_series('UMCSENT', observation_start=start_date, observation_end=end_date)
        icsa = fred.get_series('ICSA', observation_start=start_date, observation_end=end_date)
        t10y2y = fred.get_series('T10Y2Y', observation_start=start_date, observation_end=end_date)
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return pd.DataFrame()
    # Fetch VIX from yfinance
    try:
        vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        vix = pd.Series(dtype=float)
    # Ensure all series are 1D
    def flatten(s):
        if hasattr(s, 'values') and hasattr(s.values, 'shape') and len(s.values.shape) > 1:
            return pd.Series(s.values.flatten(), index=s.index)
        return s
    umcsent = flatten(umcsent)
    icsa = flatten(icsa)
    t10y2y = flatten(t10y2y)
    vix = flatten(vix)
    # Align all series by date
    df = pd.DataFrame({
        'UMCSENT': umcsent,
        'ICSA': icsa,
        'T10Y2Y': t10y2y,
        'VIX': vix
    })
    # Invert ICSA and VIX (lower is better)
    df['ICSA_inv'] = -df['ICSA']
    df['VIX_inv'] = -df['VIX']
    # Normalize
    norm_func = zscore if use_zscore else minmax
    df['UMCSENT_norm'] = norm_func(df['UMCSENT'])
    df['ICSA_norm'] = norm_func(df['ICSA_inv'])
    df['T10Y2Y_norm'] = norm_func(df['T10Y2Y'])
    df['VIX_norm'] = norm_func(df['VIX_inv'])
    # Composite score (simple average)
    df['MacroSentimentScore'] = df[['UMCSENT_norm', 'ICSA_norm', 'T10Y2Y_norm', 'VIX_norm']].mean(axis=1)
    # Return only date and score
    return df[['MacroSentimentScore']].dropna()

# --- Sector-Specific Sentiment Function ---
def get_sector_sentiment(ticker, start_date=None, end_date=None, use_zscore=True):
    """
    Fetch and combine sector-specific sentiment indicators for a given ticker.
    Returns a DataFrame indexed by date with SectorSentimentScore.
    """
    if end_date is None:
        end_date = datetime.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365*2)
    # Get sector from yfinance
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', None)
    except Exception as e:
        print(f"Error fetching sector for {ticker}: {e}")
        return pd.DataFrame()
    fred = Fred(api_key=FRED_API_KEY)
    indicators = {}
    # Helper to safely fetch and validate FRED series
    def safe_get_series(series_id, invert=False):
        try:
            s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            if s is not None and len(s.dropna()) > 0:
                return -s if invert else s
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
        return None

    # Select indicators based on sector
    if sector == 'Real Estate':
        mortgage = safe_get_series('MORTGAGE30US', invert=True)
        houst = safe_get_series('HOUST')
        if mortgage is not None:
            indicators['MORTGAGE30US_inv'] = mortgage
        if houst is not None:
            indicators['HOUST'] = houst
    elif sector == 'Technology':
        ipn_info = safe_get_series('IPN51100N')
        indpro = safe_get_series('INDPRO')
        if ipn_info is not None:
            indicators['IPN51100N'] = ipn_info
        if indpro is not None:
            indicators['INDPRO'] = indpro
    elif sector == 'Consumer Cyclical':
        rsafs = safe_get_series('RSAFS')
        concconf = safe_get_series('CONCCONF')
        if rsafs is not None:
            indicators['RSAFS'] = rsafs
        if concconf is not None:
            indicators['CONCCONF'] = concconf
    elif sector == 'Industrials':
        indpro = safe_get_series('INDPRO')  # Industrial Production Index
        cumfnfb = safe_get_series('CUMFNFB')  # Capacity Utilization: Manufacturing
        pcu333 = safe_get_series('PCU333')  # PPI: Machinery Manufacturing
        if indpro is not None:
            indicators['INDPRO'] = indpro
        if cumfnfb is not None:
            indicators['CUMFNFB'] = cumfnfb
        if pcu333 is not None:
            indicators['PCU333'] = pcu333
    elif sector == 'Communication Services':
        # Alternative indicators for Communication Services
        ces_info = safe_get_series('CES5000000001')  # All Employees, Information
        ppi_telecom = safe_get_series('PCU51731173')  # PPI Wired Telecom
        if ces_info is not None:
            indicators['CES5000000001'] = ces_info
        if ppi_telecom is not None:
            indicators['PCU51731173'] = ppi_telecom
    elif sector == 'Healthcare':
        mcpi = safe_get_series('MCPI')  # Consumer Price Index for Medical Care
        cpihossl = safe_get_series('CPIHOSSL')  # CPI for Hospital Services
        cpihcsl = safe_get_series('CPIHCSL')  # CPI for Health Care Services
        if mcpi is not None:
            indicators['MCPI'] = mcpi
        if cpihossl is not None:
            indicators['CPIHOSSL'] = cpihossl
        if cpihcsl is not None:
            indicators['CPIHCSL'] = cpihcsl
    elif sector == 'Healthcare':
        ces_health = safe_get_series('CES6000000001')  # All Employees, Health Care
        hlth_index = safe_get_series('HLTH')  # Health Care Price Index
        ppi_health = safe_get_series('PCU621')  # PPI Ambulatory Health Care Services
        if ces_health is not None:
            indicators['CES6000000001'] = ces_health
        if hlth_index is not None:
            indicators['HLTH'] = hlth_index
        if ppi_health is not None:
            indicators['PCU621'] = ppi_health
    elif sector is None:
        # Handle ETFs and tickers with no sector: use broad market indicators
        print(f"Sector is None. Using broad market indicators for ETF or index.")
        sp500 = safe_get_series('SP500')  # S&P 500 Index
        will5000 = safe_get_series('WILL5000PR')  # Wilshire 5000 Total Market Index
        if sp500 is not None:
            indicators['SP500'] = sp500
        if will5000 is not None:
            indicators['WILL5000PR'] = will5000
    else:
        print(f"Sector '{sector}' not supported for sector sentiment.")

    # If no valid indicators, return NaN DataFrame and warn
    if not indicators:
        print(f"[WARN] No valid sector indicators found for sector '{sector}'. Returning NaN sector sentiment.")
        nan_df = pd.DataFrame({'SectorSentimentScore': [np.nan]}, index=[datetime.today()])
        return nan_df
    # Build DataFrame
    df = pd.DataFrame(indicators)
    # Normalize
    norm_func = zscore if use_zscore else minmax
    norm_cols = {}
    for col in df.columns:
        norm_cols[col + '_norm'] = norm_func(df[col])
    for col, normed in norm_cols.items():
        df[col] = normed
    # Composite score
    df['SectorSentimentScore'] = df[[c for c in df.columns if c.endswith('_norm')]].mean(axis=1)
    # Forward fill and always include today's date
    result = df[['SectorSentimentScore']].dropna().copy()
    today = pd.Timestamp(datetime.today().date())
    if today not in result.index:
        # Add today's date with last available score
        last_score = result['SectorSentimentScore'].iloc[-1] if not result.empty else np.nan
        result.loc[today] = last_score
    result = result.sort_index()
    return result

# --- Example Usage ---
if __name__ == "__main__":
    # Simple CLI for sector sentiment testing
    default_ticker = 'OPEN'
    ticker = input(f"Enter stock ticker for sector sentiment (default: {default_ticker}): ").strip().upper() or default_ticker
    sector_df = get_sector_sentiment(ticker)
    # Get sector info for display
    try:
        sector = yf.Ticker(ticker).info.get('sector', None)
    except Exception:
        sector = None
    print(f"\n[INFO] Ticker: {ticker}")
    print(f"[INFO] Detected Sector: {sector}")
    print("[INFO] Sector Sentiment Score (last 5):\n", sector_df.tail())
