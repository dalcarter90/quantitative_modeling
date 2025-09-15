"""
Hybrid Pipeline: ARIMAX + GARCH-X + LSTM
Runs ARIMAX and GARCH-X, saves their outputs, and uses them as features for LSTM.
"""

import numpy as np
import pandas as pd
from arima_forecast import ARIMAXForecaster
from garch_model import GARCHForecaster
from lstm_model import LSTMForecaster
from data_loader import StockDataLoader

def run_hybrid_pipeline(forecast_steps=30):
    # Print TensorFlow GPU info
    from lstm_model import print_tf_gpu_info
    print_tf_gpu_info()
    # --- 4. Create Features for LSTM (FIX #3 - No Data Leakage) ---
    # We need the HISTORICAL, IN-SAMPLE predictions from ARIMAX, not the future forecast.
    # The .fittedvalues attribute gives us this.
    # (This block moved below after ARIMAX and GARCH are defined)

    # --- User Input Section (No changes here) ---
    ticker = input("Enter stock ticker (default: OPEN): ").strip().upper()
    if not ticker:
        ticker = "OPEN"
    period_options = ["2y", "1y", "6mo", "90d", "60d", "30d"]
    print("Select lookback period:")
    for i, opt in enumerate(period_options, 1):
        print(f"  {i}. {opt}")
    period_choice = input(f"Enter choice (1-{len(period_options)}, default 1): ").strip()
    try:
        period_idx = int(period_choice) - 1
        period = period_options[period_idx] if 0 <= period_idx < len(period_options) else period_options[0]
    except Exception:
        period = period_options[0]
    print(f"\n[INFO] Using ticker: {ticker}, period: {period}")

    # --- 1. Load Data ---
    loader = StockDataLoader()
    stock_data = loader.fetch_stock_data(ticker, period=period)
    if stock_data is None or len(stock_data) < 100:
        raise ValueError(f"Insufficient {ticker} stock data")
    close_prices = stock_data['Close']
    # Remove timezone from index for alignment with ARIMAX and other features
    if hasattr(close_prices.index, 'tz') and close_prices.index.tz is not None:
        close_prices.index = close_prices.index.tz_localize(None)

    # --- 2. Run GARCH-X on Percentage Returns (FIX #1) ---
    # GARCH must be trained on percentage returns to be valid.
    returns = 100 * close_prices.pct_change().dropna()
    # --- Section 2 of your pipeline: Run GARCH-X Correctly ---
    print("INFO: Preparing data for GARCH model...")
    # 1. Instantiate the forecaster (simple GARCH for now)
    garch = GARCHForecaster(model_type='GARCH', p=1, q=1)

    # 2. Use the .prepare_returns() method. This is the most critical step.
    garch.prepare_returns(close_prices)

    # 3. Fit the model using the correctly prepared internal data.
    print("INFO: Fitting GARCH model...")
    garch.fit()

    # 4. Get the correctly scaled volatility feature
    garch_vol_feature = garch.fitted_values
    print("INFO: GARCH analysis complete. Volatility feature created.")

    # --- 3. Run ARIMAX with GARCH Volatility as a Feature (FIX #2) ---
    arimax = ARIMAXForecaster(ticker)
    # Prepare the GARCH volatility as an exogenous feature for ARIMAX
    # Ensure it's aligned to the main price index
    arimax_exog = pd.DataFrame({
        'garch_vol': garch_vol_feature
    }).reindex(close_prices.index).fillna(method='ffill').fillna(method='bfill')

    # Set exogenous data on the ARIMAX forecaster, then run analysis
    arimax.exog = arimax_exog
    arimax.run_complete_analysis(
        period=period, 
        forecast_steps=forecast_steps
    )

    # --- 4. Create Features for LSTM (FIX #3 - No Data Leakage) ---
    # We need the HISTORICAL, IN-SAMPLE predictions from ARIMAX, not the future forecast.
    # The .fittedvalues attribute gives us this.
    arimax_pred_feature = arimax.fitted_model.fittedvalues
    print("\n[DEBUG] ARIMAX fittedvalues head:")
    print(arimax_pred_feature.head())
    print("[DEBUG] ARIMAX fittedvalues tail:")
    print(arimax_pred_feature.tail())
    print(f"[DEBUG] ARIMAX fittedvalues NaN count: {arimax_pred_feature.isnull().sum()}")
    print(f"[DEBUG] ARIMAX fittedvalues index[0]: {arimax_pred_feature.index[0] if len(arimax_pred_feature) else 'N/A'}")

    # Reindex all features to close_prices.index
    arimax_pred_feature = arimax_pred_feature.reindex(close_prices.index)
    garch_vol_feature = garch_vol_feature.reindex(close_prices.index, method='ffill').fillna(method='bfill')
    print("[DEBUG] After reindexing:")
    print(f"arimax_pred_feature head:\n{arimax_pred_feature.head()}")
    print(f"garch_vol_feature head:\n{garch_vol_feature.head()}")
    print(f"arimax_pred_feature NaN count: {arimax_pred_feature.isnull().sum()}")
    print(f"garch_vol_feature NaN count: {garch_vol_feature.isnull().sum()}")

    # --- 4. Create Features for LSTM (FIX #3 - No Data Leakage) ---
    # We need the HISTORICAL, IN-SAMPLE predictions from ARIMAX, not the future forecast.
    # The .fittedvalues attribute gives us this.
    arimax_pred_feature = arimax.fitted_model.fittedvalues

    # Combine all features into a single DataFrame
    print("\n[DEBUG] Lengths and index alignment of features:")
    print(f"close_prices: {len(close_prices)}, index[0]: {close_prices.index[0] if len(close_prices) else 'N/A'}")
    features = pd.DataFrame(index=close_prices.index)
    features['price'] = close_prices
    features['arimax_pred'] = arimax_pred_feature
    features['garch_vol'] = garch_vol_feature

    print("\n[DEBUG] Features shape after creation:", features.shape)
    print("[DEBUG] Features head:")
    print(features.head())
    print("[DEBUG] Features tail:")
    print(features.tail())
    print("[DEBUG] NaN count per column before cleaning:")
    print(features.isnull().sum())
    print("[DEBUG] Inf count per column before cleaning:")
    print(np.isinf(features).sum())

    # Clean the combined features: forward-fill then back-fill
    features.fillna(method='ffill', inplace=True)
    features.fillna(method='bfill', inplace=True)
    print("[DEBUG] NaN count per column after ffill/bfill:")
    print(features.isnull().sum())
    print("[DEBUG] Inf count per column after ffill/bfill:")
    print(np.isinf(features).sum())

    # Replace inf/-inf with NaN, then drop any remaining NaN rows
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("[DEBUG] NaN count per column after replacing inf with nan:")
    print(features.isnull().sum())
    n_before = len(features)
    features.dropna(inplace=True)
    n_after = len(features)
    if n_after < n_before:
        print(f"[WARNING] Dropped {n_before - n_after} rows with NaN or Inf values from features before LSTM.")
    print("\n--- Feature DataFrame for LSTM ---")
    print(features.head())
    print(features.tail())
    print("[DEBUG] Final features shape:", features.shape)
    print("Missing values check:")
    print(features.isnull().sum())
    print("-" * 30)

    # --- 5. Save features and Train LSTM ---
    features.to_csv(f"results/{ticker}_hybrid_features.csv")
    print(f"Saved hybrid features to results/{ticker}_hybrid_features.csv")
    
    lstm = LSTMForecaster(sequence_length=20, epochs=100)
    
    # Ensure 'price' is the first column for LSTM target
    if 'price' in features.columns and features.columns[0] != 'price':
        features = features[['price'] + [col for col in features.columns if col != 'price']]
    X_train, y_train, X_test, y_test, _, _ = lstm.prepare_data(features)
    
    print(f"LSTM input shape: {X_train.shape}")
    print(f"LSTM test input shape: {X_test.shape}")

    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("Training or test set is empty after preparation.")

    lstm.build_model((X_train.shape[1], X_train.shape[2]))
    lstm.train(X_train, y_train, X_val=X_test, y_val=y_test)
    
    # --- 6. Evaluate ---
    metrics = lstm.evaluate(X_test, y_test)
    print("\nLSTM Hybrid Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # --- 7. Multi-step Price Prediction ---
    # Recursive prediction for 5, 10, 30, 60, 90 days
    forecast_horizons = [1, 5, 10, 30, 60, 90]
    scaled_full = lstm.scaler.transform(features.values)
    last_seq = scaled_full[-lstm.sequence_length:].copy()
    num_features = features.shape[1]
    forecasts_scaled = []
    seq = last_seq.copy()
    for step in range(1, max(forecast_horizons)+1):
        input_seq = seq.reshape(1, lstm.sequence_length, num_features)
        scaled_pred = lstm.model.predict(input_seq)
        # Prepare next input: shift left, append new pred as price, keep other features as last known
        next_row = seq[-1].copy()
        next_row[0] = scaled_pred[0, 0]
        seq = np.vstack([seq[1:], next_row])
        forecasts_scaled.append(next_row.copy())

    # Inverse transform only the predicted price, not the full feature vector
    forecasts_scaled = np.array(forecasts_scaled)  # shape (max_horizon, num_features)
    # For each forecasted row, inverse-transform only the price (first column)
    prices_unscaled = []
    for row in forecasts_scaled:
        # Create a dummy row for inverse_transform: set price to predicted, others to 0
        dummy = np.zeros_like(row)
        dummy[0] = row[0]
        price = lstm.scaler.inverse_transform(dummy.reshape(1, -1))[0, 0]
        prices_unscaled.append(price)

    print("\n==============================")
    print("Forecast Summary (LSTM Hybrid):")
    for idx, h in enumerate(forecast_horizons):
        price = prices_unscaled[h-1]
        if h == 1:
            print(f"Next-day predicted price: ${price:.2f}")
        else:
            print(f"{h}-day predicted price: ${price:.2f}")
    print("==============================")

if __name__ == "__main__":
    run_hybrid_pipeline()
