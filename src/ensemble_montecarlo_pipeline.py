import optuna
# --- Hyperparameter Tuning Utilities ---
def tune_xgboost(features, target_col, forecast_steps, n_trials=20):
    train_features = features.iloc[:-forecast_steps]
    test_features = features.iloc[-forecast_steps:]
    X_train = train_features.drop(columns=[target_col])
    y_train = train_features[target_col]
    X_test = test_features.drop(columns=[target_col])
    y_test = test_features[target_col]
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'reg:squarederror',
            'verbosity': 0
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        return rmse
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    return preds, best_model, best_params

def tune_lightgbm(features, target_col, forecast_steps, n_trials=20):
    train_features = features.iloc[:-forecast_steps]
    test_features = features.iloc[-forecast_steps:]
    X_train = train_features.drop(columns=[target_col])
    y_train = train_features[target_col]
    X_test = test_features.drop(columns=[target_col])
    y_test = test_features[target_col]
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        return rmse
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_model = LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    return preds, best_model, best_params
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# --- XGBoost Forecasting ---
def forecast_xgboost(features, target_col, forecast_steps):
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, objective='reg:squarederror', verbosity=0)
    train_features = features.iloc[:-forecast_steps]
    test_features = features.iloc[-forecast_steps:]
    X_train = train_features.drop(columns=[target_col])
    y_train = train_features[target_col]
    model.fit(X_train, y_train)
    preds = model.predict(test_features.drop(columns=[target_col]))
    return preds, model

# --- LightGBM Forecasting ---
def forecast_lightgbm(features, target_col, forecast_steps):
    model = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    train_features = features.iloc[:-forecast_steps]
    test_features = features.iloc[-forecast_steps:]
    X_train = train_features.drop(columns=[target_col])
    y_train = train_features[target_col]
    model.fit(X_train, y_train)
    preds = model.predict(test_features.drop(columns=[target_col]))
    return preds, model
import json
import os
# --- Exogenous Variable Loader ---
def load_exogenous_variables(ticker, period=None, json_path=None):
    """
    Loads and formats exogenous variables for a given ticker from smart_ml_persistent_data.json.
    Optionally filter by period (not implemented, but can be extended).
    Returns a pandas DataFrame of features indexed by timestamp.
    """
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), '..', 'smart_ml_persistent_data.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    stock_data = []
    for entry in data.get('learning_data', {}).get('stocks', []):
        stock_features = entry.get('data', {}).get(ticker)
        if stock_features:
            # Remove non-feature keys
            features = {k: v for k, v in stock_features.items() if k not in ['timestamp', 'data_points', 'mode']}
            features['timestamp'] = stock_features.get('timestamp')
            stock_data.append(features)
    if not stock_data:
        return None
    df = pd.DataFrame(stock_data)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

"""
Ensemble Forecasting + Monte Carlo Simulation Pipeline
Implements a five-stage pipeline for robust probabilistic financial forecasting.
"""

# Force matplotlib to use a non-GUI backend to avoid Tkinter errors in threads
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from arima_forecast import ARIMAXForecaster
from garch_model import GARCHForecaster
from lstm_model import LSTMForecaster
from data_loader import StockDataLoader
from random_forest_model import RandomForestForecaster
from explainability import ExplainabilityModule
from evaluation import evaluation_report
from scenario_stress import scenario_stress_test
from sklearn.linear_model import LinearRegression
import shap
from scipy.stats import norm
# Sentiment analysis imports
from sentiment_analysis import get_macro_sentiment, get_sector_sentiment

# --- Stage 1: Data Preparation ---
def prepare_data(ticker, period, exog_vars=None):
    loader = StockDataLoader()
    stock_data = loader.fetch_stock_data(ticker, period=period)
    if stock_data is None or len(stock_data) < 100:
        raise ValueError(f"Insufficient {ticker} stock data")
    close_prices = stock_data['Close']
    if hasattr(close_prices.index, 'tz') and close_prices.index.tz is not None:
        close_prices.index = close_prices.index.tz_localize(None)
    log_returns = np.log(close_prices).diff().dropna()
    # Load exogenous variables if not provided
    if exog_vars is None:
        exog_vars = load_exogenous_variables(ticker)
    return close_prices, log_returns, exog_vars

# --- Stage 2: Parallel Forecasting ---
def forecast_garch(log_returns, forecast_steps):
    # Use GARCHForecaster with model_type='GARCH', p=1, o=1, q=1 for GJR-GARCH
    garch = GARCHForecaster(model_type='GARCH', p=1, o=1, q=1)
    garch.prepare_returns(np.exp(log_returns))
    garch.fit()
    # Manual pause after model summary
    print("[INFO] GARCH model summary complete. Press Enter to continue...")
    input()
    result = garch.forecast_volatility(horizon=forecast_steps)
    forecast_vol = result['volatility_forecast']
    return forecast_vol, garch

def forecast_arimax(log_returns, exog, forecast_steps, ticker, period):
    arimax = ARIMAXForecaster(ticker)
    # Use robust preparation logic from ARIMAXForecaster
    arimax.load_and_prepare_data(period=period, price_column='Close', use_technical_enhancement=True)
    arimax.find_best_arima_params()
    arimax.fit_model()
    forecast_result = arimax.forecast(steps=forecast_steps)
    if isinstance(forecast_result, dict) and 'forecast' in forecast_result:
        forecast_mu = forecast_result['forecast'].values
    else:
        forecast_mu = np.asarray(forecast_result)
    return forecast_mu, arimax

def forecast_lstm(features, forecast_steps):
    lstm = LSTMForecaster(sequence_length=20, epochs=100)
    X_train, y_train, X_test, y_test, _, _ = lstm.prepare_data(features)
    lstm.build_model((X_train.shape[1], X_train.shape[2]))
    lstm.train(X_train, y_train, X_val=X_test, y_val=y_test)
    scaled_full = lstm.scaler.transform(features.values)
    last_seq = scaled_full[-lstm.sequence_length:]
    forecast_mu = lstm.forecast(last_seq, steps=forecast_steps)
    return forecast_mu, lstm

# --- Random Forest Forecasting ---
def forecast_random_forest(features, target_col, forecast_steps):
    rf = RandomForestForecaster()
    # Use all but last forecast_steps for training, last for prediction
    train_features = features.iloc[:-forecast_steps]
    test_features = features.iloc[-forecast_steps:]
    rf.fit(train_features, target_col)
    preds = rf.predict(test_features.drop(columns=[target_col]))
    return preds, rf

# --- Stage 3: Parameter Synthesis & Annualization ---
# --- Stacking Ensemble ---
def stacking_ensemble(base_preds, y_true=None):
    # base_preds: dict of model_name -> prediction array
    X = np.column_stack([base_preds[k] for k in sorted(base_preds)])
    meta = LinearRegression()
    # If y_true is provided, fit meta-model; else, just predict
    if y_true is not None:
        meta.fit(X, y_true)
        final_pred = meta.predict(X)
    else:
        final_pred = np.mean(X, axis=1)  # fallback to mean if no y_true
    return final_pred, meta

# --- Hybrid ARIMA-GARCH: returns both mean and volatility forecasts ---
def hybrid_arima_garch(log_returns, forecast_steps, ticker, period):
    arima_mu, arima_model = forecast_arimax(log_returns, None, forecast_steps, ticker, period)
    arima_resid = arima_model.fitted_model.resid if hasattr(arima_model, 'fitted_model') else None
    if arima_resid is None:
        raise ValueError("ARIMA model did not produce residuals.")
    # Use GARCHForecaster with model_type='GARCH', p=1, o=1, q=1 for GJR-GARCH
    garch = GARCHForecaster(model_type='GARCH', p=1, o=1, q=1)
    garch.prepare_returns(pd.Series(arima_resid))
    garch.fit()
    garch_vol = garch.forecast_volatility(horizon=forecast_steps)['volatility_forecast']
    return arima_mu, garch_vol, garch

# --- Stage 4: Monte Carlo Simulation ---
def monte_carlo_sim(mu, sigma, S0, n_sims=5000, n_days=30, model='GBM'):
    dt = 1/252
    S = np.zeros((n_sims, n_days+1))
    S[:,0] = S0
    for t in range(1, n_days+1):
        Z = np.random.normal(size=n_sims)
        S[:,t] = S[:,t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return S

# --- Stage 5: Analysis and Reporting ---
def analyze_simulation(S, strike=None):
    final_prices = S[:,-1]
    mean = np.mean(final_prices)
    median = np.median(final_prices)
    std = np.std(final_prices)
    p5 = np.percentile(final_prices, 5)
    p95 = np.percentile(final_prices, 95)
    VaR_95 = np.percentile(final_prices, 5)
    VaR_99 = np.percentile(final_prices, 1)
    ES_95 = final_prices[final_prices <= VaR_95].mean()
    ES_99 = final_prices[final_prices <= VaR_99].mean()
    report = {
        'mean': mean,
        'median': median,
        'std': std,
        '5th_percentile': p5,
        '95th_percentile': p95,
        'VaR_95': VaR_95,
        'VaR_99': VaR_99,
        'ES_95': ES_95,
        'ES_99': ES_99,
    }
    # Option pricing with risk-free rate discounting
    # Default risk-free rate (annualized)
    risk_free_rate = 0.05  # 5% annualized, can be parameterized
    n_days = S.shape[1] - 1
    T = n_days / 252  # time to maturity in years
    discount_factor = np.exp(-risk_free_rate * T)
    if strike:
        call_payoff = np.maximum(final_prices - strike, 0)
        put_payoff = np.maximum(strike - final_prices, 0)
        call_price = np.mean(call_payoff) * discount_factor
        put_price = np.mean(put_payoff) * discount_factor
        report['call_option_price'] = call_price
        report['put_option_price'] = put_price
        report['risk_free_rate'] = risk_free_rate
        report['discount_factor'] = discount_factor
    return report


def run_ensemble_pipeline(
    ticker='OPEN', period='1y', forecast_steps=30, n_sims=5000, strike=None,
    scenario_mu=None, scenario_sigma=None
):
    # Stage 1
    close_prices, log_returns, exog_vars = prepare_data(ticker, period)
    close_prices = close_prices.dropna()
    if len(close_prices) == 0:
        raise ValueError("No valid close prices after dropping NaNs.")
    # --- Sentiment Analysis ---
    macro_sentiment_df = get_macro_sentiment()
    sector_sentiment_df = get_sector_sentiment(ticker)
    sentiment_df = pd.merge(macro_sentiment_df, sector_sentiment_df, left_index=True, right_index=True, how='outer').ffill()
    # Optionally, print or log the latest sentiment scores
    print("[SENTIMENT] Latest MacroSentimentScore:", macro_sentiment_df.tail(1).to_dict())
    print(f"[SENTIMENT] Latest SectorSentimentScore for {ticker}:", sector_sentiment_df.tail(1).to_dict())
    S0 = close_prices.iloc[-1]
    print(f"[DEBUG] S0 (last close price): {S0}")
    if S0 == 0 or np.isnan(S0):
        raise ValueError(f"S0 (last close price) is zero or NaN: {S0}")

    # Debug: Check log_returns scaling and distribution
    print(f"[DEBUG] log_returns (first 5): {log_returns.head().values}")
    print(f"[DEBUG] log_returns (last 5): {log_returns.tail().values}")
    print(f"[DEBUG] log_returns mean: {log_returns.mean()}, std: {log_returns.std()}, min: {log_returns.min()}, max: {log_returns.max()}")
    print(f"[DEBUG] forecast_steps: {forecast_steps}")

    # Stage 2: Model Forecasts
    with ThreadPoolExecutor() as executor:
        # Hybrid ARIMA-GARCH
        print(f"[DEBUG] Submitting hybrid_arima_garch with forecast_steps={forecast_steps}")
        hybrid_future = executor.submit(hybrid_arima_garch, log_returns, forecast_steps, ticker, period)
        # LSTM
        features = pd.DataFrame({'log_return': log_returns})
        lstm_future = executor.submit(forecast_lstm, features, forecast_steps)
        # Random Forest (use lagged returns as features)
        rf_features = features.copy()
        for lag in range(1, 6):
            rf_features[f'lag_{lag}'] = rf_features['log_return'].shift(lag)
        rf_features = rf_features.dropna()
        rf_future = executor.submit(forecast_random_forest, rf_features.assign(target=rf_features['log_return']), 'target', forecast_steps)
        # XGBoost (Optuna tuning)
        xgb_future = executor.submit(tune_xgboost, rf_features.assign(target=rf_features['log_return']), 'target', forecast_steps, 20)
        # LightGBM (Optuna tuning)
        lgbm_future = executor.submit(tune_lightgbm, rf_features.assign(target=rf_features['log_return']), 'target', forecast_steps, 20)
        # Wait for results
        hybrid_result = hybrid_future.result()
        if isinstance(hybrid_result, tuple) and len(hybrid_result) == 3:
            arima_mu, garch_vol, garch_obj = hybrid_result
        else:
            raise RuntimeError("hybrid_arima_garch must return (arima_mu, garch_vol, garch_obj)")
        (lstm_mu, lstm_model) = lstm_future.result()
        (rf_mu, rf_model) = rf_future.result()
        (xgb_mu, xgb_model, xgb_params) = xgb_future.result()
        (lgbm_mu, lgbm_model, lgbm_params) = lgbm_future.result()
        # If garch_obj is not set, re-create it for simulation parameters
        if garch_obj is None:
            garch_obj = GARCHForecaster(model_type='GARCH', p=1, o=1, q=1)
            garch_obj.prepare_returns(pd.Series(arima_mu))
            garch_obj.fit()

    # Debug: GARCH volatility output for this horizon
    print(f"[DEBUG] GARCH volatility (first 5): {np.array(garch_vol)[:5]}")
    print(f"[DEBUG] GARCH volatility (last 5): {np.array(garch_vol)[-5:]}")
    print(f"[DEBUG] GARCH volatility mean: {np.mean(garch_vol)}, std: {np.std(garch_vol)}, min: {np.min(garch_vol)}, max: {np.max(garch_vol)}")

    # Align lengths for stacking
    min_len = min(len(arima_mu), len(lstm_mu), len(rf_mu), len(xgb_mu), len(lgbm_mu))
    base_preds = {
        'arima': np.array(arima_mu)[-min_len:],
        'lstm': np.array(lstm_mu)[-min_len:],
        'rf': np.array(rf_mu)[-min_len:],
        'xgb': np.array(xgb_mu)[-min_len:],
        'lgbm': np.array(lgbm_mu)[-min_len:]
    }
    # For meta-model, use last min_len true values
    y_true = features['log_return'].values[-min_len:]
    stacked_pred, meta_model = stacking_ensemble(base_preds, y_true)


    # Stage 3: Use GARCH model's simulation parameters for Monte Carlo, with safety checks
    garch_fit_significant = True
    garch_fit_failure_reason = None
    mu_annual = 0.0  # Default drift is 0 for safety
    sigma_annual = None
    # Try to validate GARCH fit
    try:
        garch_fit = getattr(garch_obj, 'fit_result', None)
        if garch_fit is not None and hasattr(garch_fit, 'pvalues'):
            pvals = garch_fit.pvalues
            # Check beta[1] significance (may be named 'beta[1]' or similar)
            beta_keys = [k for k in pvals.keys() if 'beta' in k]
            beta1_key = None
            if beta_keys:
                # Prefer 'beta[1]' if present, else first beta
                beta1_key = 'beta[1]' if 'beta[1]' in beta_keys else beta_keys[0]
            if beta1_key and pvals[beta1_key] >= 0.10:
                garch_fit_significant = False
                garch_fit_failure_reason = f"GARCH beta[1] p-value not significant: {pvals[beta1_key]:.3f}"
        # If fit is significant, use GARCH params
        if garch_fit_significant:
            sim_params = garch_obj.get_simulation_parameters()
            mu_annual = 0.0  # Always use 0 drift for safety
            sigma_annual = sim_params['sigma']
        else:
            raise ValueError(garch_fit_failure_reason or "GARCH fit not significant")
    except Exception as e:
        # Fallback: use historical volatility, mu=0
        sigma_annual = np.std(log_returns) * np.sqrt(252)
        garch_fit_failure_reason = garch_fit_failure_reason or str(e)
        print(f"[WARN] GARCH Fit Failure: {garch_fit_failure_reason}. Falling back to historical volatility.")

    print(f"[DEBUG] (Monte Carlo) mu_annual: {mu_annual}, sigma_annual: {sigma_annual}, S0: {S0}")

    # Convert annualized mu and sigma to daily values for simulation
    mu_daily = mu_annual / 252
    sigma_daily = sigma_annual / np.sqrt(252)
    print(f"[DEBUG] (Monte Carlo) mu_daily: {mu_daily}, sigma_daily: {sigma_daily}")

    # --- Volatility Governor ---

    # --- Dynamic Volatility Cap for Short Horizons ---
    fallback_used = False
    orig_sigma_daily = sigma_daily
    recent_vols = log_returns[-10:] if len(log_returns) >= 10 else log_returns
    realized_vol_recent = recent_vols.std() if len(recent_vols) > 0 else None
    dynamic_cap = None
    if forecast_steps <= 7 and realized_vol_recent is not None:
        dynamic_cap = 1.5 * np.max(np.abs(recent_vols))
        print(f"[INFO] Dynamic volatility cap for {forecast_steps}d horizon: {dynamic_cap:.4f}")
        if sigma_daily > dynamic_cap:
            sigma_daily = min(sigma_daily, dynamic_cap)
            fallback_used = True
            print(f"[INFO] Applied dynamic cap: sigma_daily set to {sigma_daily:.4f}")
    else:
        VOL_CAP = 0.05  # fallback for longer horizons
        if sigma_daily > VOL_CAP:
            print(f"[WARN] GARCH sigma_daily {sigma_daily:.4f} exceeds cap of {VOL_CAP:.2%}. Using realized volatility as fallback.")
            realized_vol_30 = log_returns[-30:].std() if len(log_returns) >= 30 else None
            realized_vol_60 = log_returns[-60:].std() if len(log_returns) >= 60 else None
            realized_vol = realized_vol_30 if realized_vol_30 is not None else realized_vol_60
            if realized_vol is not None:
                sigma_daily = min(realized_vol, VOL_CAP)
                fallback_used = True
                print(f"[INFO] Using realized volatility: {sigma_daily:.4f} (capped at {VOL_CAP:.2%})")
            else:
                sigma_daily = VOL_CAP
                print(f"[INFO] Not enough data for realized volatility. Using cap: {VOL_CAP:.2%}")

    # Dynamic horizon adjustment if volatility is still too high (unchanged logic)
    VOL_CRITICAL = 0.05
    sim_horizon = forecast_steps
    if sigma_daily > VOL_CRITICAL:
        sim_horizon = min(forecast_steps, 5)
        print(f"[WARN] Volatility {sigma_daily:.2%} > {VOL_CRITICAL:.2%}. Reducing simulation horizon to {sim_horizon} days.")

    # Stage 4: Monte Carlo Simulation (with scenario override)
    if scenario_mu is not None or scenario_sigma is not None:
        # If scenario overrides are provided, assume they are in annualized terms and convert
        custom_mu = scenario_mu / 252 if scenario_mu is not None else None
        custom_sigma = scenario_sigma / np.sqrt(252) if scenario_sigma is not None else None
        S = scenario_stress_test(S0, n_sims=n_sims, n_days=sim_horizon, mu=mu_daily, sigma=sigma_daily, custom_mu=custom_mu, custom_sigma=custom_sigma)
    else:
        S = monte_carlo_sim(mu_daily, sigma_daily, S0, n_sims=n_sims, n_days=sim_horizon)

    # Stage 5: Evaluation
    eval_metrics = evaluation_report(y_true, stacked_pred)
    report = analyze_simulation(S, strike=strike)
    report['RMSE'] = eval_metrics['RMSE']
    report['MAE'] = eval_metrics['MAE']
    # Add recent realized volatility info
    report['recent_realized_volatility'] = float(realized_vol_recent) if realized_vol_recent is not None else None
    report['dynamic_volatility_cap'] = float(dynamic_cap) if dynamic_cap is not None else None
    # Add per-model forecast results to the report
    report['model_forecasts'] = {
        'arima': np.array(arima_mu)[-min_len:].tolist(),
        'lstm': np.array(lstm_mu)[-min_len:].tolist(),
        'rf': np.array(rf_mu)[-min_len:].tolist(),
        'xgb': np.array(xgb_mu)[-min_len:].tolist(),
        'lgbm': np.array(lgbm_mu)[-min_len:].tolist(),
        'stacked': stacked_pred.tolist()
    }
    print(f"[EVAL] RMSE: {eval_metrics['RMSE']:.6f}, MAE: {eval_metrics['MAE']:.6f}")

    # Add sentiment scores to report
    report['sentiment'] = {
        'macro': macro_sentiment_df.tail(1).to_dict(),
        'sector': sector_sentiment_df.tail(1).to_dict(),
        'merged': sentiment_df.tail(1).to_dict()
    }

    # Stage 6: Explainability (SHAP)
    try:
        explainer = ExplainabilityModule(rf_model.model, model_type='tree')
        # Use last min_len rows for SHAP, drop 'target' only if present
        drop_cols = [c for c in ['target'] if c in rf_features.columns]
        X_shap = rf_features.drop(columns=drop_cols).values[-min_len:]
        shap_values = explainer.explain(X_shap, plot=False)
        report['shap_values'] = shap_values
    except Exception as e:
        print(f"[WARN] SHAP explainability failed: {e}")
        report['shap_values'] = None

    return report

if __name__ == "__main__":
    # User input for ticker and lookback period
    default_ticker = 'OPEN'
    period_options = ['2y', '1y', '6mo', '90d', '60d', '30d']
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

    # Prediction timeframe selection
    timeframe_options = [
        ("1 day", 1),
        ("3 days", 3),
        ("1 week", 5),
        ("30 days", 30),
        ("60 days", 60),
        ("90 days", 90),
        ("6 months", 126)  # ~21 trading days per month
    ]
    print("\nSelect prediction timeframe:")
    for i, (label, _) in enumerate(timeframe_options, 1):
        print(f"  {i}. {label}")
    timeframe_choice = input(f"Enter choice (1-{len(timeframe_options)}, default 2): ").strip()
    try:
        timeframe_idx = int(timeframe_choice) - 1
        forecast_steps = timeframe_options[timeframe_idx][1] if 0 <= timeframe_idx < len(timeframe_options) else timeframe_options[1][1]
    except Exception:
        forecast_steps = timeframe_options[1][1]

    print(f"\n[INFO] Using ticker: {ticker}, period: {period}, prediction horizon: {forecast_steps} days")
    result = run_ensemble_pipeline(ticker=ticker, period=period, forecast_steps=forecast_steps)
    print("\n===== Ensemble Monte Carlo Report =====")
    for k, v in result.items():
        if k == 'model_forecasts':
            print("\n--- Individual Model Forecasts (last horizon) ---")
            for model, vals in v.items():
                print(f"{model}: {vals}")
        elif k == 'sentiment':
            print("\n--- Sentiment Scores ---")
            print("MacroSentimentScore:", v['macro'])
            print("SectorSentimentScore:", v['sector'])
            print("Merged Sentiment DataFrame (latest):", v['merged'])
        else:
            print(f"{k}: {v}")
