"""
Enhanced technical indicators module for ARIMA forecasting.
This module provides technical analysis indicators that can improve ARIMA input data.
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Technical analysis indicators for stock price enhancement."""
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        pass
    
    def calculate_ema(self, series, window=20):
        """Calculate Exponential Moving Average.
        
        Args:
            series (pd.Series): Price series
            window (int): EMA period
            
        Returns:
            pd.Series: EMA values
        """
        return series.ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, series, window=14):
        """Calculate Relative Strength Index.
        
        Args:
            series (pd.Series): Price series
            window (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic_rsi(self, series, window=14, smooth_k=3, smooth_d=3):
        """Calculate Stochastic RSI.
        
        Args:
            series (pd.Series): Price series
            window (int): RSI period
            smooth_k (int): %K smoothing
            smooth_d (int): %D smoothing
            
        Returns:
            tuple: (stoch_rsi, %K, %D)
        """
        # First calculate RSI
        rsi = self.calculate_rsi(series, window)
        
        # Calculate Stochastic of RSI
        rsi_min = rsi.rolling(window=window).min()
        rsi_max = rsi.rolling(window=window).max()
        
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        
        # Smooth %K and calculate %D
        k_percent = stoch_rsi.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        
        return stoch_rsi, k_percent, d_percent
    
    def calculate_bollinger_bands(self, series, window=20, num_std=2):
        """Calculate Bollinger Bands.
        
        Args:
            series (pd.Series): Price series
            window (int): Moving average period
            num_std (float): Number of standard deviations
            
        Returns:
            tuple: (middle_band, upper_band, lower_band)
        """
        middle_band = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return middle_band, upper_band, lower_band
    
    def detect_regime_changes(self, series, short_window=20, long_window=50):
        """Detect market regime changes using moving average crossovers.
        
        Args:
            series (pd.Series): Price series
            short_window (int): Short-term MA period
            long_window (int): Long-term MA period
            
        Returns:
            pd.Series: Regime signals (1=bullish, -1=bearish, 0=neutral)
        """
        short_ma = series.rolling(window=short_window).mean()
        long_ma = series.rolling(window=long_window).mean()
        
        # Create regime signals
        regime = pd.Series(0, index=series.index)
        regime[short_ma > long_ma] = 1  # Bullish
        regime[short_ma < long_ma] = -1  # Bearish
        
        return regime
    
    def enhance_price_series(self, data, price_column='Close'):
        """Enhance price series using technical indicators for better ARIMA input.
        
        This doesn't add indicators to ARIMA directly, but uses them to:
        1. Detect and handle outliers
        2. Identify regime changes
        3. Apply appropriate transformations
        
        Args:
            data (pd.DataFrame): Stock data
            price_column (str): Price column to enhance
            
        Returns:
            dict: Enhanced data and analysis
        """
        series = data[price_column].copy()
        
        # Calculate indicators
        ema_20 = self.calculate_ema(series, 20)
        ema_50 = self.calculate_ema(series, 50)
        rsi = self.calculate_rsi(series)
        stoch_rsi, k_pct, d_pct = self.calculate_stochastic_rsi(series)
        bb_middle, bb_upper, bb_lower = self.calculate_bollinger_bands(series)
        regime = self.detect_regime_changes(series)
        
        # Identify potential outliers using Bollinger Bands
        bb_outliers = (series > bb_upper) | (series < bb_lower)
        
        # Identify extreme RSI conditions
        oversold = rsi < 30
        overbought = rsi > 70
        
        # Create enhanced series with outlier handling
        enhanced_series = series.copy()
        
        # Optional: Smooth extreme movements (conservative approach)
        # You can uncomment this for more conservative outlier handling
        # extreme_moves = np.abs(series.pct_change()) > 0.1  # 10% daily moves
        # if extreme_moves.any():
        #     enhanced_series[extreme_moves] = ema_20[extreme_moves]
        
        enhancement_info = {
            'original_series': series,
            'enhanced_series': enhanced_series,
            'technical_indicators': {
                'ema_20': ema_20,
                'ema_50': ema_50,
                'rsi': rsi,
                'stoch_rsi': stoch_rsi,
                'stoch_k': k_pct,
                'stoch_d': d_pct,
                'bb_middle': bb_middle,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'regime': regime
            },
            'market_conditions': {
                'current_regime': regime.iloc[-1] if len(regime) > 0 else 0,
                'recent_oversold_periods': oversold.tail(30).sum(),
                'recent_overbought_periods': overbought.tail(30).sum(),
                'bb_outlier_frequency': bb_outliers.sum() / len(series) * 100,
                'trend_strength': abs(ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1] * 100
            }
        }
        
        print(f"Technical Analysis Summary:")
        print(f"Current regime: {'Bullish' if regime.iloc[-1] > 0 else 'Bearish' if regime.iloc[-1] < 0 else 'Neutral'}")
        print(f"Current RSI: {rsi.iloc[-1]:.1f}")
        print(f"Current Stochastic RSI: {stoch_rsi.iloc[-1]:.1f}")
        print(f"Trend strength: {enhancement_info['market_conditions']['trend_strength']:.1f}%")
        print(f"BB outlier frequency: {enhancement_info['market_conditions']['bb_outlier_frequency']:.1f}%")
        
        return enhancement_info


def get_multivariate_features(data, price_column='Close'):
    """Get features for multivariate time series analysis (alternative to pure ARIMA).
    
    This creates a feature matrix that could be used with ARIMAX or machine learning models.
    
    Args:
        data (pd.DataFrame): Stock data
        price_column (str): Price column
        
    Returns:
        pd.DataFrame: Feature matrix
    """
    tech_calc = TechnicalIndicators()
    series = data[price_column]
    
    # Price-based features
    features = pd.DataFrame(index=data.index)
    features['price'] = series
    features['returns'] = series.pct_change()
    features['log_returns'] = np.log(series / series.shift(1))
    
    # Volume features (if available)
    if 'Volume' in data.columns:
        features['volume'] = data['Volume']
        features['volume_ma'] = data['Volume'].rolling(20).mean()
        features['price_volume_ratio'] = series / data['Volume']
    
    # Technical indicators
    features['ema_20'] = tech_calc.calculate_ema(series, 20)
    features['ema_50'] = tech_calc.calculate_ema(series, 50)
    features['rsi'] = tech_calc.calculate_rsi(series)
    
    stoch_rsi, k_pct, d_pct = tech_calc.calculate_stochastic_rsi(series)
    features['stoch_rsi'] = stoch_rsi
    features['stoch_k'] = k_pct
    features['stoch_d'] = d_pct
    
    bb_middle, bb_upper, bb_lower = tech_calc.calculate_bollinger_bands(series)
    features['bb_position'] = (series - bb_lower) / (bb_upper - bb_lower)
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        features[f'price_lag_{lag}'] = series.shift(lag)
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
    
    # Market regime
    features['regime'] = tech_calc.detect_regime_changes(series)
    
    return features.dropna()


if __name__ == "__main__":
    # Example usage
    print("Technical Indicators module loaded successfully!")
    print("This module can enhance ARIMA forecasting through:")
    print("1. Enhanced data preprocessing")
    print("2. Outlier detection and handling")
    print("3. Market regime identification")
    print("4. Feature engineering for multivariate models")
