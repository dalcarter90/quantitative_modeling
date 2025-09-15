import pandas as pd
import yfinance as yf

def fetch_implied_volatility(ticker, strike, expiry, option_type='call'):
    """
    Fetches the implied volatility for a given option from Yahoo Finance using yfinance.
    Returns IV as a decimal (e.g., 0.45 for 45%) or None if not found.
    """
    try:
        stock = yf.Ticker(ticker)
        options = stock.options
        if not options:
            print(f"No options expiries found for {ticker}.")
            return None
        # Find closest expiry
        expiry = min(options, key=lambda x: abs((pd.Timestamp(x) - pd.Timestamp(expiry)).days))
        opt_chain = stock.option_chain(expiry)
        chain = opt_chain.calls if option_type == 'call' else opt_chain.puts
        # Find closest strike
        chain['strike_diff'] = (chain['strike'] - strike).abs()
        row = chain.loc[chain['strike_diff'].idxmin()]
        iv = row.get('impliedVolatility', None)
        if iv is not None:
            print(f"[INFO] Fetched implied volatility for {ticker} {option_type} {strike} {expiry}: {iv:.4f}")
        return iv
    except Exception as e:
        print(f"[WARN] Could not fetch IV: {e}")
        return None
"""
Black-Scholes Option Pricing Comparison Script
Compares Monte Carlo option prices from the ensemble pipeline to Black-Scholes analytical prices.
"""
import numpy as np
from scipy.stats import norm
from ensemble_montecarlo_pipeline import run_ensemble_pipeline

def black_scholes_price(S0, K, T, sigma, r=0.05, option_type='call'):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

def black_scholes_greeks(S0, K, T, sigma, r=0.05, option_type='call'):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S0 * nd1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        theta = (-S0 * nd1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    gamma = nd1 / (S0 * sigma * np.sqrt(T))
    vega = S0 * nd1 * np.sqrt(T) / 100
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }
    """
    S0: initial stock price
    K: strike price
    T: time to maturity (years)
    sigma: annualized volatility
    r: risk-free rate (annualized)
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

if __name__ == "__main__":
    # Interactive selection for ticker, lookback, and horizon
    default_ticker = 'OPEN'
    period_options = ['2y', '1y', '6mo', '90d', '60d', '30d']
    timeframe_options = [
        ("1 day", 1),
        ("3 days", 3),
        ("1 week", 5),
        ("30 days", 30),
        ("60 days", 60),
        ("90 days", 90),
        ("6 months", 126)
    ]

    ticker = input(f"Enter stock ticker (default: {default_ticker}): ").strip().upper() or default_ticker
    print("Select lookback period:")
    for i, opt in enumerate(period_options, 1):
        print(f"  {i}. {opt}")
    period_choice = input(f"Enter choice (1-{len(period_options)}, default 2): ").strip()
    try:
        period_idx = int(period_choice) - 1
        period = period_options[period_idx] if 0 <= period_idx < len(period_options) else period_options[1]
    except Exception:
        period = period_options[1]

    print("\nSelect prediction horizon:")
    for i, (label, _) in enumerate(timeframe_options, 1):
        print(f"  {i}. {label}")
    timeframe_choice = input(f"Enter choice (1-{len(timeframe_options)}, default 3): ").strip()
    try:
        timeframe_idx = int(timeframe_choice) - 1
        forecast_steps = timeframe_options[timeframe_idx][1] if 0 <= timeframe_idx < len(timeframe_options) else timeframe_options[2][1]
    except Exception:
        forecast_steps = timeframe_options[2][1]

    n_sims = 5000
    # Run pipeline to get MC prices and parameters
    result = run_ensemble_pipeline(ticker=ticker, period=period, forecast_steps=forecast_steps, n_sims=n_sims, strike=None)
    S0 = result['mean']
    strike = S0
    sigma_annual = None
    if 'std' in result:
        sigma_annual = result['std'] * np.sqrt(252)
    else:
        sigma_annual = 0.3  # fallback
    T = forecast_steps / 252
    r = result.get('risk_free_rate', 0.05)

    # Prompt for expiry (default to nearest available)
    import datetime
    today = datetime.date.today()
    # Use yfinance to get available expiries
    stock = yf.Ticker(ticker)
    expiries = stock.options
    expiry = None
    if expiries:
        print("\nAvailable option expiries:")
        for i, exp in enumerate(expiries, 1):
            print(f"  {i}. {exp}")
        expiry_choice = input(f"Select expiry (1-{len(expiries)}, default 1): ").strip()
        try:
            expiry_idx = int(expiry_choice) - 1
            expiry = expiries[expiry_idx] if 0 <= expiry_idx < len(expiries) else expiries[0]
        except Exception:
            expiry = expiries[0]
    else:
        expiry = (today + datetime.timedelta(days=forecast_steps)).strftime('%Y-%m-%d')

    # Fetch real-time implied volatility
    iv_call = fetch_implied_volatility(ticker, strike, expiry, option_type='call')
    iv_put = fetch_implied_volatility(ticker, strike, expiry, option_type='put')

    # Use IV if available, else fallback to pipeline volatility
    sigma_call = iv_call if iv_call is not None else sigma_annual
    sigma_put = iv_put if iv_put is not None else sigma_annual

    # Black-Scholes prices
    bs_call = black_scholes_price(S0, strike, T, sigma_call, r, option_type='call')
    bs_put = black_scholes_price(S0, strike, T, sigma_put, r, option_type='put')
    # Monte Carlo prices
    mc_call = result.get('call_option_price', None)
    mc_put = result.get('put_option_price', None)
    print(f"\n===== Option Pricing Comparison =====")
    print(f"Underlying: {ticker}, Period: {period}, Horizon: {forecast_steps} days")
    print(f"S0 (mean simulated): {S0:.2f}, Strike: {strike:.2f}, Volatility (annualized, call): {sigma_call:.4f}, Volatility (put): {sigma_put:.4f}, Risk-free rate: {r:.2%}")
    print(f"Black-Scholes Call Price: {bs_call:.4f}")
    print(f"Monte Carlo Call Price:   {mc_call:.4f}" if mc_call is not None else "Monte Carlo Call Price:   N/A")
    print(f"Black-Scholes Put Price:  {bs_put:.4f}")
    print(f"Monte Carlo Put Price:    {mc_put:.4f}" if mc_put is not None else "Monte Carlo Put Price:    N/A")

    # Greeks calculation
    call_greeks = black_scholes_greeks(S0, strike, T, sigma_call, r, option_type='call')
    put_greeks = black_scholes_greeks(S0, strike, T, sigma_put, r, option_type='put')
    print("\n--- Black-Scholes Greeks (Call) ---")
    for k, v in call_greeks.items():
        print(f"{k.capitalize()}: {v:.6f}")
    print("\n--- Black-Scholes Greeks (Put) ---")
    for k, v in put_greeks.items():
        print(f"{k.capitalize()}: {v:.6f}")
