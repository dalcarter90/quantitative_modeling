def print_tf_gpu_info():
    import tensorflow as tf
    print("[TensorFlow] Built with CUDA:", tf.test.is_built_with_cuda())
    print("[TensorFlow] GPU devices:", tf.config.list_physical_devices('GPU'))
    if tf.config.list_physical_devices('GPU'):
        print("[TensorFlow] Using GPU for training.")
    else:
        print("[TensorFlow] No GPU detected. Training will use CPU.")
"""
LSTM Model for Stock Price Forecasting
=====================================

This module implements Long Short-Term Memory (LSTM) neural networks
for time series forecasting of stock prices.

Author: AI Assistant
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class LSTMForecaster:
    """
    LSTM-based forecasting model for stock prices.
    
    Attributes:
        sequence_length (int): Number of time steps to look back
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        model (Sequential): Trained LSTM model
        scaler (MinMaxScaler): Data scaler
        history (dict): Training history
    """
    
    def __init__(self, sequence_length=60, epochs=100, batch_size=32):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length (int): Number of time steps for sequences
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def prepare_data(self, data, train_size=0.8):
        """
        Prepare data for LSTM training.
        
        Args:
            data (pd.Series or np.array): Time series data
            train_size (float): Proportion of data for training
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, train_data, test_data)
        """
        # Accept DataFrame or Series or ndarray
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            data = data.reshape(-1, 1)
        # Check for NaNs or Infs before scaling
        print(f"[DEBUG] Data shape before scaling: {data.shape}")
        print(f"[DEBUG] Data sample before scaling: {data[:5]}")
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"[DEBUG] NaN count: {np.isnan(data).sum()}, Inf count: {np.isinf(data).sum()}")
            raise ValueError("NaN or Inf detected in input features before scaling.")
            return  # Ensure function exits after error
        # Check for constant columns
        if data.shape[1] > 1:
            nunique = (pd.DataFrame(data).nunique() == 1)
            if nunique.any():
                print("Warning: One or more features are constant. This can cause issues with MinMaxScaler.")
        # Split into train and test (before scaling)
        n_train = int(len(data) * train_size)
        train_data_raw = data[:n_train]
        test_data_raw = data[n_train:]
        # Fit scaler only on training data
        try:
            self.scaler.fit(train_data_raw)
            train_data = self.scaler.transform(train_data_raw)
            test_data = self.scaler.transform(test_data_raw)
            scaled_data = np.vstack([train_data, test_data])
        except Exception as e:
            raise RuntimeError(f"Scaling failed: {e}\nData sample: {data[:5]}")
        # Create sequences (multivariate)
        X_train, y_train = self._create_sequences(train_data)
        X_test, y_test = self._create_sequences(test_data)
        return X_train, y_train, X_test, y_test, train_data, test_data
    
    def _create_sequences(self, data):
        """
        Create sequences for LSTM training.
        
        Args:
            data (np.array): Scaled time series data
            
        Returns:
            tuple: (X, y) sequences
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, :])
            y.append(data[i, 0])  # Predict price (first column)
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input data
        """
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the LSTM model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array, optional): Validation features
            y_val (np.array, optional): Validation targets
        """
        # Reshape input to be 3D [samples, time steps, features] only if univariate
        if X_train.ndim == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        if X_val is not None:
            if X_val.ndim == 2:
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Build model if not already built
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2] if X_train.ndim == 3 else 1))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (np.array): Input sequences
            
        Returns:
            np.array: Predictions (scaled)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Reshape input if needed
        if X.ndim == 2:
            # Univariate: (samples, timesteps) -> (samples, timesteps, 1)
            X = X.reshape((X.shape[0], X.shape[1], 1))
        elif X.ndim == 1:
            # Single sequence: (timesteps,) -> (1, timesteps, 1)
            X = X.reshape((1, X.shape[0], 1))
        # If already 3D, do nothing
        predictions = self.model.predict(X)
        return predictions
    
    def forecast(self, last_sequence, steps=30):
        """
        Generate multi-step ahead forecasts.
        
        Args:
            last_sequence (np.array): Last sequence from training data
            steps (int): Number of steps to forecast
            
        Returns:
            np.array: Forecasted values (original scale)
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Ensure last_sequence is properly scaled
        if last_sequence.ndim == 1:
            last_sequence = last_sequence.reshape(-1, 1)
        
        # Use the last sequence_length points
        current_sequence = last_sequence[-self.sequence_length:].copy()
        predictions = []
        
        for _ in range(steps):
            # Reshape for prediction
            X = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Predict next value
            next_pred = self.model.predict(X, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence (remove first, add prediction)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0, 0]
        
        # Convert back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_scaled = self.scaler.inverse_transform(predictions)
        
        return predictions_scaled.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        predictions = self.predict(X_test)
        # Inverse transform to original scale
        y_test_scaled = y_test.reshape(-1, 1)
        predictions_scaled = predictions.reshape(-1, 1)
        # For multivariate, only inverse transform the first column (price)
        if self.scaler.n_features_in_ > 1:
            dummy = np.zeros((len(y_test_scaled), self.scaler.n_features_in_))
            dummy[:, 0] = y_test_scaled[:, 0]
            y_test_orig = self.scaler.inverse_transform(dummy)[:, 0]
            dummy[:, 0] = predictions_scaled[:, 0]
            predictions_orig = self.scaler.inverse_transform(dummy)[:, 0]
        else:
            y_test_orig = self.scaler.inverse_transform(y_test_scaled)
            predictions_orig = self.scaler.inverse_transform(predictions_scaled)

        # Debug: Check for NaNs
        def nan_report(arr, name):
            print(f"{name}: shape={arr.shape}, nan_count={np.isnan(arr).sum()}, min={np.nanmin(arr)}, max={np.nanmax(arr)}")
        nan_report(y_test_orig, 'y_test_orig')
        nan_report(predictions_orig, 'predictions_orig')
        if np.isnan(y_test_orig).any() or np.isnan(predictions_orig).any():
            raise ValueError("NaN detected in y_test_orig or predictions_orig. Check data and model output.")

        # Calculate metrics
        mse = mean_squared_error(y_test_orig, predictions_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, predictions_orig)
        mape = np.mean(np.abs((y_test_orig - predictions_orig) / y_test_orig)) * 100
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, actual, predictions, title="LSTM Predictions vs Actual"):
        """
        Plot predictions vs actual values.
        
        Args:
            actual (np.array): Actual values
            predictions (np.array): Predicted values
            title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    """
    Main function to demonstrate LSTM forecasting with OPEN stock.
    """
    print("LSTM Model Demo - Stock Forecasting")
    print("====================================")

    # User input for ticker and lookback period
    ticker = input("Enter stock ticker (default: OPEN): ").strip().upper()
    if not ticker:
        ticker = "OPEN"
    period_options = ["30d", "60d", "90d", "6mo", "1y", "2y"]
    print("Select lookback period:")
    for i, opt in enumerate(period_options, 1):
        print(f"  {i}. {opt}")
    period_choice = input(f"Enter choice (1-{len(period_options)}, default 6): ").strip()
    try:
        period_idx = int(period_choice) - 1
        period = period_options[period_idx] if 0 <= period_idx < len(period_options) else period_options[-1]
    except Exception:
        period = period_options[-1]
    print(f"\n[INFO] Using ticker: {ticker}, period: {period}")

    # Load real stock data
    try:
        from data_loader import StockDataLoader

        data_loader = StockDataLoader()
        stock_data = data_loader.fetch_stock_data(ticker, period=period)

        if stock_data is None or len(stock_data) < 100:
            raise ValueError(f"Insufficient {ticker} stock data")

        print(f"Loaded {len(stock_data)} days of {ticker} stock data")
        print(f"Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
        print(f"Price range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")

        # Use Close prices for LSTM training
        data = stock_data['Close']

        # Initialize and train LSTM model
        lstm_model = LSTMForecaster(sequence_length=60, epochs=100)

        print(f"Preparing {ticker} stock data...")
        try:
            X_train, y_train, X_test, y_test, _, _ = lstm_model.prepare_data(data)
        except ValueError as ve:
            print(f"Data preparation error: {ve}")
            return
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        print("Training LSTM model on OPEN stock...")
        print("⚠️  This may take several minutes...")
        lstm_model.train(X_train, y_train, X_test, y_test)
        
        print("Evaluating OPEN stock model...")
        metrics = lstm_model.evaluate(X_test, y_test)
        
        print("\nOPEN Stock LSTM Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nGenerating OPEN stock forecast...")
        last_sequence = data.values[-lstm_model.sequence_length:]
        forecast = lstm_model.forecast(last_sequence, steps=30)
        
        current_price = data.iloc[-1]
        print(f"Current OPEN price: ${current_price:.2f}")
        print(f"30-day LSTM forecast: ${forecast[0]:.2f}, ${forecast[4]:.2f}, ${forecast[9]:.2f}... (days 1, 5, 10)")
        print(f"30-day price change: {((forecast[-1] - current_price) / current_price * 100):+.2f}%")
        
    except Exception as e:
        print(f"Error loading OPEN data: {e}")
        print("Falling back to simulated data...")
        
        # Fallback to original simulated data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        prices = 100 + np.cumsum(np.random.randn(200) * 0.1)
        
        data = pd.Series(prices, index=dates)
        
        # Initialize and train LSTM model
        lstm_model = LSTMForecaster(sequence_length=30, epochs=50)
        
        print("Preparing data...")
        try:
            X_train, y_train, X_test, y_test, _, _ = lstm_model.prepare_data(data)
        except ValueError as ve:
            print(f"Data preparation error: {ve}")
            return
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        print("Training LSTM model...")
        lstm_model.train(X_train, y_train, X_test, y_test)
        
        print("Evaluating model...")
        metrics = lstm_model.evaluate(X_test, y_test)
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("Generating forecast...")
        last_sequence = data.values[-lstm_model.sequence_length:]
        forecast = lstm_model.forecast(last_sequence, steps=30)
        
        print(f"30-day forecast: {forecast[:5]}... (showing first 5 values)")

if __name__ == "__main__":
    main()
