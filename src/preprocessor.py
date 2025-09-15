"""
Data preprocessing module for time series analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class TimeSeriesPreprocessor:
    """Class to handle time series preprocessing for ARIMA modeling."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    def extract_price_series(self, data, price_column='Close'):
        """Extract price series from stock data.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            price_column (str): Column to use for analysis ('Open', 'High', 'Low', 'Close')
            
        Returns:
            pd.Series: Price time series
        """
        if price_column not in data.columns:
            raise ValueError(f"Column '{price_column}' not found in data")
        
        price_series = data[price_column].copy()
        
        # Remove any missing values
        price_series = price_series.dropna()
        
        # Sort by date
        price_series = price_series.sort_index()
        
        print(f"Extracted {len(price_series)} price observations")
        print(f"Date range: {price_series.index.min()} to {price_series.index.max()}")
        
        return price_series
    
    def check_stationarity(self, series, alpha=0.05):
        """Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series (pd.Series): Time series to test
            alpha (float): Significance level
            
        Returns:
            dict: Test results
        """
        result = adfuller(series.dropna())
        
        is_stationary = result[1] <= alpha
        
        adf_results = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': is_stationary,
            'confidence_level': (1 - alpha) * 100
        }
        
        print(f"ADF Test Results:")
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.6f}")
        
        if is_stationary:
            print(f"Series is stationary at {(1-alpha)*100}% confidence level")
        else:
            print(f"Series is NOT stationary at {(1-alpha)*100}% confidence level")
        
        return adf_results
    
    def difference_series(self, series, order=1):
        """Apply differencing to make series stationary.
        
        Args:
            series (pd.Series): Time series to difference
            order (int): Order of differencing
            
        Returns:
            pd.Series: Differenced series
        """
        diff_series = series.copy()
        
        for i in range(order):
            diff_series = diff_series.diff().dropna()
            print(f"Applied differencing order {i+1}")
        
        print(f"Differenced series length: {len(diff_series)}")
        return diff_series
    
    def log_transform(self, series):
        """Apply log transformation to stabilize variance.
        
        Args:
            series (pd.Series): Time series to transform
            
        Returns:
            pd.Series: Log-transformed series
        """
        if (series <= 0).any():
            print("Warning: Series contains non-positive values. Adding constant.")
            series = series + abs(series.min()) + 1
        
        log_series = np.log(series)
        print("Applied log transformation")
        return log_series
    
    def decompose_series(self, series, model='additive', period=None):
        """Decompose time series into trend, seasonal, and residual components.
        
        Args:
            series (pd.Series): Time series to decompose
            model (str): 'additive' or 'multiplicative'
            period (int): Period for seasonal decomposition
            
        Returns:
            statsmodels.tsa.seasonal.DecomposeResult: Decomposition results
        """
        if period is None:
            # Try to infer period based on data frequency
            if len(series) >= 365:
                period = 252  # Trading days in a year
            else:
                period = min(30, len(series) // 4)  # Monthly or quarterly
        
        try:
            decomposition = seasonal_decompose(series, model=model, period=period)
            print(f"Series decomposed using {model} model with period {period}")
            return decomposition
        except Exception as e:
            print(f"Error in decomposition: {str(e)}")
            return None
    
    def remove_outliers(self, series, method='iqr', threshold=3):
        """Remove outliers from time series.
        
        Args:
            series (pd.Series): Time series
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.Series: Series with outliers removed
        """
        original_length = len(series)
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_series = series[(series >= lower_bound) & (series <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            cleaned_series = series[z_scores < threshold]
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        removed_count = original_length - len(cleaned_series)
        print(f"Removed {removed_count} outliers using {method} method")
        
        return cleaned_series
    
    def prepare_for_arima(self, series, target_column='Close', max_diff_order=2):
        """Prepare time series for ARIMA modeling.
        
        Args:
            series (pd.Series or pd.DataFrame): Time series data
            target_column (str): Target column if DataFrame is provided
            max_diff_order (int): Maximum differencing order to try
            
        Returns:
            dict: Prepared data and transformation info
        """
        # Extract series if DataFrame is provided
        if isinstance(series, pd.DataFrame):
            series = self.extract_price_series(series, target_column)
        
        original_series = series.copy()
        
        # Check for missing values
        if series.isnull().any():
            print("Handling missing values...")
            series = series.dropna()
        
        # Apply log transformation if series has high variance
        log_transformed = False
        if series.std() / series.mean() > 0.5:  # High coefficient of variation
            print("High variance detected, applying log transformation...")
            series = self.log_transform(series)
            log_transformed = True
        
        # Test stationarity and apply differencing if needed
        stationarity_results = []
        diff_order = 0
        current_series = series.copy()
        
        for order in range(max_diff_order + 1):
            adf_result = self.check_stationarity(current_series)
            stationarity_results.append(adf_result)
            
            if adf_result['is_stationary']:
                diff_order = order
                break
            
            if order < max_diff_order:
                current_series = self.difference_series(current_series, 1)
        
        final_series = current_series
        
        preparation_info = {
            'original_series': original_series,
            'final_series': final_series,
            'log_transformed': log_transformed,
            'diff_order': diff_order,
            'stationarity_tests': stationarity_results,
            'is_stationary': stationarity_results[-1]['is_stationary']
        }
        
        print(f"\nPreparation complete:")
        print(f"- Log transformed: {log_transformed}")
        print(f"- Differencing order: {diff_order}")
        print(f"- Final series is stationary: {preparation_info['is_stationary']}")
        print(f"- Final series length: {len(final_series)}")
        
        return preparation_info


if __name__ == "__main__":
    # Example usage
    from data_loader import StockDataLoader
    
    loader = StockDataLoader()
    data = loader.fetch_stock_data("OPEN", period="2y")
    
    if data is not None:
        preprocessor = TimeSeriesPreprocessor()
        prep_info = preprocessor.prepare_for_arima(data)
        
        print(f"Original series shape: {prep_info['original_series'].shape}")
        print(f"Final series shape: {prep_info['final_series'].shape}")
