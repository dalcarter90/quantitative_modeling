"""
Main ARIMAX forecasting module for stock price prediction with exogenous variables.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA  # ARIMAX is ARIMA with exog argument
from statsmodels.tsa.stattools import arma_order_select_ic
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_loader import StockDataLoader
from preprocessor import TimeSeriesPreprocessor  
from visualizer import TimeSeriesVisualizer
from technical_indicators import TechnicalIndicators
from news_sentiment import NewsSentimentAnalyzer


class ARIMAXForecaster:
    """Class to build and use ARIMAX models for stock price forecasting with exogenous variables."""
    
    def __init__(self, ticker='RDDT'):
        """Initialize the ARIMAX forecaster.
        Args:
            ticker (str): Stock ticker symbol
        """
        self.ticker = ticker
        self.model = None
        self.fitted_model = None
        self.preparation_info = None
        self.best_params = None
        self.exog = None  # Exogenous variables for ARIMAX
        # Initialize components
        self.data_loader = StockDataLoader()
        self.preprocessor = TimeSeriesPreprocessor()
        self.visualizer = TimeSeriesVisualizer()
        self.tech_indicators = TechnicalIndicators()
        self.sentiment_analyzer = NewsSentimentAnalyzer(ticker)
    
    def validate_lookback_period(self, data_length):
        """Validate if lookback period meets statistical requirements.
        
        Based on research recommendations for financial time series.
        
        Args:
            data_length (int): Number of observations
            
        Returns:
            tuple: (quality_rating, recommendation)
        """
        recommendations = {
            'minimum_viable': 100,
            'statistically_robust': 250,  # ~1 year of trading days
            'highly_reliable': 500,       # ~2 years of trading days
            'diminishing_returns': 1000   # ~4 years of trading days
        }
        
        if data_length < recommendations['minimum_viable']:
            return "INSUFFICIENT", "Need at least 100 observations for basic ARIMA modeling"
        elif data_length < recommendations['statistically_robust']:
            return "MINIMAL", "Consider more data for robust parameter estimates"
        elif data_length < recommendations['highly_reliable']:
            return "GOOD", "Adequate for reliable forecasting (1-year optimal)"
        elif data_length < recommendations['diminishing_returns']:
            return "EXCELLENT", "Optimal balance of data quantity and structural stability"
        else:
            return "EXCESSIVE", "Consider testing for structural breaks - may include outdated patterns"
    
    def check_data_stability(self, series):
        """Check for potential structural breaks in the time series.
        
        Simple statistical tests for data stability over the lookback period.
        
        Args:
            series (pd.Series): Price time series
            
        Returns:
            dict: Stability analysis results
        """
        # Split data into halves for comparison
        mid_point = len(series) // 2
        first_half = series[:mid_point]
        second_half = series[mid_point:]
        
        # Calculate basic statistics for each half
        first_mean = first_half.mean()
        second_mean = second_half.mean()
        first_std = first_half.std()
        second_std = second_half.std()
        
        # Calculate percentage changes
        mean_change = abs(second_mean - first_mean) / first_mean * 100
        volatility_change = abs(second_std - first_std) / first_std * 100
        
        # Simple stability assessment
        is_stable = mean_change < 30 and volatility_change < 50  # Thresholds based on typical market behavior
        
        stability_info = {
            'is_stable': is_stable,
            'mean_change_pct': mean_change,
            'volatility_change_pct': volatility_change,
            'first_half_mean': first_mean,
            'second_half_mean': second_mean,
            'assessment': 'STABLE' if is_stable else 'POTENTIAL_STRUCTURAL_CHANGE'
        }
        
        print(f"\nData Stability Analysis:")
        print(f"Assessment: {stability_info['assessment']}")
        print(f"Mean price change between periods: {mean_change:.1f}%")
        print(f"Volatility change between periods: {volatility_change:.1f}%")
        
        if not is_stable:
            print("⚠️  Warning: Potential structural changes detected. Consider shorter lookback period.")
        else:
            print("✅ Data appears stable over the lookback period.")
        
        return stability_info
    
    def load_and_prepare_data(self, period='1y', price_column='Close', use_technical_enhancement=True):
        """Load and prepare stock data for ARIMA modeling.
        
        Args:
            period (str): Time period for data retrieval
            price_column (str): Price column to use for forecasting
            use_technical_enhancement (bool): Whether to use technical indicators for data enhancement
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"Loading data for {self.ticker}...")
        
        # Load data
        self.raw_data = self.data_loader.fetch_stock_data(self.ticker, period=period)
        if self.raw_data is None:
            print("Failed to load data")
            return False
        
        # Get stock info
        stock_info = self.data_loader.get_stock_info(self.ticker)
        if stock_info:
            print(f"Stock: {stock_info['name']}")
            print(f"Sector: {stock_info['sector']}")
        
        # Validate data quality and lookback period
        quality_rating, recommendation = self.validate_lookback_period(len(self.raw_data))
        print(f"\nData Quality Assessment:")
        print(f"Observations: {len(self.raw_data)}")
        print(f"Quality Rating: {quality_rating}")
        print(f"Recommendation: {recommendation}")
        
        # Trading days calculation for context
        trading_days_per_year = 252
        years_of_data = len(self.raw_data) / trading_days_per_year
        print(f"Approximate years of data: {years_of_data:.1f}")
        
        # Visualize raw data
        self.visualizer.plot_price_series(self.raw_data, price_column, self.ticker)
        
        # Optional: Enhance data using technical indicators
        if use_technical_enhancement:
            print(f"\nApplying technical analysis enhancement...")
            self.enhancement_info = self.tech_indicators.enhance_price_series(
                self.raw_data, price_column
            )
            
            # Use enhanced series for ARIMA if significant improvements detected
            volatility_reduction = (
                self.enhancement_info['original_series'].std() - 
                self.enhancement_info['enhanced_series'].std()
            ) / self.enhancement_info['original_series'].std() * 100
            
            if volatility_reduction > 2:  # If volatility reduced by >2%
                print(f"✅ Using technical-enhanced series (volatility reduced by {volatility_reduction:.1f}%)")
                enhanced_data = self.raw_data.copy()
                enhanced_data[price_column] = self.enhancement_info['enhanced_series']
                data_for_arima = enhanced_data
            else:
                print(f"Technical enhancement didn't provide significant improvement (volatility reduction: {volatility_reduction:.1f}%)")
                data_for_arima = self.raw_data
        else:
            data_for_arima = self.raw_data
        
        # Add sentiment analysis enhancement
        try:
            print(f"\n📰 Adding news sentiment analysis...")
            sentiment_enhanced_data = self.sentiment_analyzer.enhance_price_data_with_sentiment(data_for_arima)
            
            # Get sentiment summary
            sentiment_summary = self.sentiment_analyzer.get_current_sentiment_summary()
            print(f"Current sentiment: {sentiment_summary.get('current_sentiment', 0):.3f}")
            print(f"7-day average sentiment: {sentiment_summary.get('avg_sentiment_7d', 0):.3f}")
            print(f"Sentiment trend: {sentiment_summary.get('sentiment_trend', 'neutral')}")
            
            data_for_arima = sentiment_enhanced_data
            self.sentiment_summary = sentiment_summary
            
        except Exception as e:
            print(f"⚠️ Sentiment analysis failed: {e}")
            print("Proceeding without sentiment features...")
            self.sentiment_summary = None
        
        # Prepare data for ARIMA
        print(f"\nPreparing data for ARIMA modeling...")
        self.preparation_info = self.preprocessor.prepare_for_arima(
            data_for_arima, price_column
        )
        
        # Check data stability
        self.stability_info = self.check_data_stability(self.preparation_info['original_series'])
        
        if not self.preparation_info['is_stationary']:
            print("Warning: Series may not be stationary after preprocessing")
        
        # Visualize stationarity transformation
        self.visualizer.plot_stationarity_test(
            self.preparation_info['original_series'],
            self.preparation_info['final_series'],
            self.ticker
        )
        
        # Plot ACF/PACF for parameter estimation
        self.visualizer.plot_acf_pacf(
            self.preparation_info['final_series'], 
            ticker=self.ticker
        )
        
        # Plot seasonal decomposition
        self.visualizer.plot_decomposition(
            self.preparation_info['original_series'], 
            ticker=self.ticker
        )
        
        return True
    
    def find_best_arima_params(self, max_p=3, max_d=2, max_q=3, ic='aic'):
        """Find best ARIMA parameters using grid search.
        
        Optimized for 1-year lookback period based on research:
        - Reduced parameter space for efficiency
        - Focus on lower orders for shorter time series
        
        Args:
            max_p (int): Maximum AR order to test (reduced from 5 to 3)
            max_d (int): Maximum differencing order to test
            max_q (int): Maximum MA order to test (reduced from 5 to 3)
            ic (str): Information criterion ('aic', 'bic', 'hqic')
            
        Returns:
            tuple: Best (p, d, q) parameters
        """
        if self.preparation_info is None:
            raise ValueError("Data must be prepared first. Call load_and_prepare_data()")
        
        print(f"Searching for best ARIMA parameters...")
        print(f"Testing p=[0,{max_p}], d=[0,{max_d}], q=[0,{max_q}]")
        
        series = self.preparation_info['final_series']
        
        # Use the differencing order from preprocessing as starting point
        best_d = self.preparation_info['diff_order']
        
        best_score = float('inf')
        best_params = None
        results = []
        
        # Grid search over p and q, with d fixed from preprocessing
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    # Test the preprocessing-determined d first
                    model = ARIMA(self.preparation_info['original_series'], 
                                order=(p, best_d, q))
                    fitted_model = model.fit()
                    
                    if ic == 'aic':
                        score = fitted_model.aic
                    elif ic == 'bic':
                        score = fitted_model.bic
                    elif ic == 'hqic':
                        score = fitted_model.hqic
                    
                    results.append((p, best_d, q, score))
                    
                    if score < best_score:
                        best_score = score
                        best_params = (p, best_d, q)
                    
                    print(f"ARIMA({p},{best_d},{q}) - {ic.upper()}: {score:.2f}")
                    
                except Exception as e:
                    print(f"ARIMA({p},{best_d},{q}) - Failed: {str(e)}")
                    continue
        
        # If preprocessing d doesn't work well, try other d values
        if best_params is None and max_d > 0:
            print("Trying alternative differencing orders...")
            for d in range(max_d + 1):
                if d == best_d:
                    continue
                for p in range(min(3, max_p + 1)):  # Limit search for efficiency
                    for q in range(min(3, max_q + 1)):
                        try:
                            model = ARIMA(self.preparation_info['original_series'], 
                                        order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if ic == 'aic':
                                score = fitted_model.aic
                            elif ic == 'bic':
                                score = fitted_model.bic
                            elif ic == 'hqic':
                                score = fitted_model.hqic
                            
                            results.append((p, d, q, score))
                            
                            if score < best_score:
                                best_score = score
                                best_params = (p, d, q)
                            
                            print(f"ARIMA({p},{d},{q}) - {ic.upper()}: {score:.2f}")
                            
                        except Exception as e:
                            continue
        
        if best_params is None:
            print("Could not find suitable ARIMA parameters")
            # Default fallback
            best_params = (1, 1, 1)
            print(f"Using default parameters: ARIMA{best_params}")
        else:
            print(f"\nBest ARIMA parameters: {best_params}")
            print(f"Best {ic.upper()} score: {best_score:.2f}")
        
        self.best_params = best_params
        return best_params
    
    def fit_model(self, order=None, exog=None):
        """Fit ARIMAX model with specified or best-found parameters and exogenous variables.
        Args:
            order (tuple): ARIMA order (p, d, q). If None, uses best found parameters.
            exog (pd.DataFrame or None): Exogenous variables for ARIMAX. If None, uses self.exog.
        Returns:
            bool: True if successful, False otherwise
        """
        if self.preparation_info is None:
            raise ValueError("Data must be prepared first. Call load_and_prepare_data()")
        if order is None:
            if self.best_params is None:
                print("Finding best parameters...")
                self.find_best_arima_params()
            order = self.best_params
        if exog is None:
            exog = self.exog
        # --- Data cleaning: drop all rows with NaNs from combined data ---
        y = self.preparation_info['original_series']
        if exog is not None:
            # --- The Correct and Final Sequence ---
            # 1. y and exog are prepared and ready
            # 2. Make BOTH indexes timezone-naive BEFORE joining
            print("Applying timezone fix...")
            y.index = y.index.tz_localize(None)
            exog.index = exog.index.tz_localize(None)
            # 3. Join the now-compatible dataframes
            print("Joining dataframes...")
            combined = y.to_frame('y').join(exog)
            # --- INSPECT NaNs BEFORE dropna ---
            print("\nFirst 20 rows of combined data (head):")
            print(combined.head(20))
            print("\nLast 20 rows of combined data (tail):")
            print(combined.tail(20))
            print("\nMissing value count per column:")
            print(combined.isnull().sum())
            # 4. Drop rows with NaNs (from technical indicators, etc.)
            print(f"Shape before fillna: {combined.shape}")
            combined.fillna(method='ffill', inplace=True)
            combined.fillna(method='bfill', inplace=True)
            print(f"Shape after fillna: {combined.shape}")
            # Now, 'combined' is ready. Check if it's empty before proceeding.
            if combined.empty:
                print("FATAL ERROR: DataFrame is empty after all cleaning steps.")
                return False
            y_clean = combined['y']
            exog_clean = combined.drop(columns=['y'])
        else:
            y_clean = y.fillna(method='ffill').fillna(method='bfill')
            exog_clean = None
        print(f"Fitting ARIMAX{order} model... (exog={'provided' if exog is not None else 'None'})")
        try:
            self.model = ARIMA(y_clean, order=order, exog=exog_clean)
            self.fitted_model = self.model.fit()
            print("Model fitted successfully!")
            print(f"\nModel Summary:")
            print(f"AIC: {self.fitted_model.aic:.2f}")
            print(f"BIC: {self.fitted_model.bic:.2f}")
            print(f"Log Likelihood: {self.fitted_model.llf:.2f}")
            # Manual pause for user to review model summary
            input("[INFO] Model summary complete. Press Enter to continue to forecasting...")
            # Analyze residuals
            residuals = self.fitted_model.resid
            self.visualizer.plot_residuals(residuals, self.ticker)
            return True
        except Exception as e:
            print(f"Error fitting model: {str(e)}")
            return False
    
    def forecast(self, steps=30, alpha=0.05):
        """Generate forecast using fitted ARIMA model.
        
        Args:
            steps (int): Number of periods to forecast
            alpha (float): Significance level for confidence intervals
            
        Returns:
            dict: Forecast results with predictions and confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first. Call fit_model()")
        
        print(f"Generating {steps}-step forecast...")
        
        # Generate forecast
        # If exogenous variables were used in fitting, handle exog for forecast
        exog_future = None
        use_exog = hasattr(self.fitted_model, 'model') and hasattr(self.fitted_model.model, 'exog') and self.fitted_model.model.exog is not None
        if use_exog:
            # Use last row of exog, repeat for forecast horizon
            last_exog = self.fitted_model.model.exog[-1]
            # If exog is a DataFrame, preserve columns
            if isinstance(self.fitted_model.model.data.orig_exog, pd.DataFrame):
                exog_future = pd.DataFrame([last_exog]*steps, columns=self.fitted_model.model.data.orig_exog.columns)
            else:
                exog_future = np.tile(last_exog, (steps, 1))
            forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha, exog=exog_future)
            # Convert to numpy array to avoid indexing issues
            if hasattr(forecast_result, 'values'):
                forecast_values = forecast_result.values
            else:
                forecast_values = np.array(forecast_result)
            # Get confidence intervals
            conf_int = self.fitted_model.get_forecast(steps=steps, alpha=alpha, exog=exog_future).conf_int()
            lower_ci = conf_int.iloc[:, 0].values
            upper_ci = conf_int.iloc[:, 1].values
        else:
            # No exog: only pass steps argument (do NOT pass alpha or exog)
            forecast_result = self.fitted_model.forecast(steps=steps)
            if hasattr(forecast_result, 'values'):
                forecast_values = forecast_result.values
            else:
                forecast_values = np.array(forecast_result)
            lower_ci = None
            upper_ci = None

        # Create forecast dates
        last_date = self.preparation_info['original_series'].index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        # Create forecast series with proper indexing
        forecast_series = pd.Series(forecast_values, index=forecast_dates)

        # Visualize forecast
        self.visualizer.plot_forecast(
            self.preparation_info['original_series'],
            forecast_series,
            confidence_intervals=(lower_ci, upper_ci),
            ticker=self.ticker,
            forecast_periods=steps
        )

        # Create interactive plot
        self.visualizer.create_interactive_forecast(
            self.preparation_info['original_series'],
            forecast_series,
            confidence_intervals=(lower_ci, upper_ci),
            ticker=self.ticker
        )

        forecast_results = {
            'forecast': forecast_series,
            'lower_ci': pd.Series(lower_ci, index=forecast_dates),
            'upper_ci': pd.Series(upper_ci, index=forecast_dates),
            'confidence_level': (1 - alpha) * 100
        }

        print(f"\nForecast Summary:")
        print(f"Next day predicted price: ${forecast_values[0]:.2f}")
        print(f"30-day average predicted price: ${np.mean(forecast_values):.2f}")
        print(f"Confidence level: {(1-alpha)*100}%")

        return forecast_results
    
    def evaluate_model(self, test_size=0.2):
        """Evaluate model performance using train-test split.
        
        Args:
            test_size (float): Fraction of data to use for testing
            
        Returns:
            dict: Evaluation metrics
        """
        if self.preparation_info is None:
            raise ValueError("Data must be prepared first. Call load_and_prepare_data()")
        
        series = self.preparation_info['original_series']
        split_point = int(len(series) * (1 - test_size))
        
        train_data = series[:split_point]
        test_data = series[split_point:]
        
        print(f"Evaluating model performance...")
        print(f"Training data: {len(train_data)} observations")
        print(f"Test data: {len(test_data)} observations")
        
        # Fit model on training data
        if self.best_params is None:
            # Use simple parameters for evaluation
            order = (1, 1, 1)
        else:
            order = self.best_params
        
        try:
            train_model = ARIMA(train_data, order=order)
            fitted_train_model = train_model.fit()

            # Generate forecasts for test period
            forecast_steps = len(test_data)
            forecast_result = fitted_train_model.forecast(steps=forecast_steps)

            # Convert forecast to numpy array if needed
            if hasattr(forecast_result, 'values'):
                forecast_values = forecast_result.values
            else:
                forecast_values = np.array(forecast_result)

            # Debug: print ARIMA order and first few values
            print(f"\nARIMA order used: {order}")
            print(f"First 5 test_data values: {test_data.values[:5]}")
            print(f"First 5 forecast values: {forecast_values[:5]}")
            print(f"Test data min/max: {test_data.values.min()}/{test_data.values.max()}")
            print(f"Forecast min/max: {forecast_values.min()}/{forecast_values.max()}")

            # Calculate metrics
            mae = mean_absolute_error(test_data.values, forecast_values)
            mse = mean_squared_error(test_data.values, forecast_values)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test_data.values - forecast_values) / test_data.values)) * 100

            # Calculate directional accuracy
            actual_direction = np.diff(test_data.values) > 0
            predicted_direction = np.diff(forecast_values) > 0
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional_accuracy
            }

            print(f"\nModel Evaluation Results:")
            print(f"Mean Absolute Error (MAE): ${mae:.2f}")
            print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"Directional Accuracy: {directional_accuracy:.2f}%")

            return metrics
            
        except Exception as e:
            print(f"Error in model evaluation: {str(e)}")
            return None
    
    def run_complete_analysis(self, period='1y', forecast_steps=30, use_learning_data=False, learning_data_path='smart_ml_persistent_data.json'):
        """Run complete ARIMAX analysis pipeline, optionally using learning data as exogenous variables.
        Args:
            period (str): Data period to retrieve (default: 1y based on research)
            forecast_steps (int): Number of periods to forecast
            use_learning_data (bool): Whether to use learning data as exogenous variables
            learning_data_path (str): Path to persistent learning data
        Returns:
            dict: Complete analysis results
        """
        print("=" * 60)
        print(f"ARIMAX STOCK PRICE FORECASTING ANALYSIS")
        print(f"Ticker: {self.ticker}")
        print(f"Period: {period} (Research-optimized lookback)")
        print("=" * 60)
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data(period=period):
            return None
        # Step 2: Find best parameters
        best_params = self.find_best_arima_params()
        # Step 2.5: Optionally load and align exogenous variables from learning data
        if use_learning_data:
            self.exog = self.load_learning_data_as_exog(learning_data_path)
        # Step 3: Fit model
        if not self.fit_model(order=best_params, exog=self.exog):
            return None
        # Step 4: Evaluate model
        evaluation_metrics = self.evaluate_model()
        # Step 5: Generate forecast
        forecast_results = self.forecast(steps=forecast_steps)
        # Compile complete results
        complete_results = {
            'ticker': self.ticker,
            'data_period': period,
            'best_arima_params': best_params,
            'model_summary': {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.llf
            },
            'evaluation_metrics': evaluation_metrics,
            'forecast_results': forecast_results,
            'sentiment_analysis': getattr(self, 'sentiment_summary', None),
            'data_info': {
                'total_observations': len(self.preparation_info['original_series']),
                'date_range': (
                    self.preparation_info['original_series'].index.min(),
                    self.preparation_info['original_series'].index.max()
                ),
                'log_transformed': self.preparation_info['log_transformed'],
                'differencing_order': self.preparation_info['diff_order'],
                'data_quality_rating': self.validate_lookback_period(len(self.preparation_info['original_series']))[0],
                'stability_assessment': self.stability_info['assessment'] if hasattr(self, 'stability_info') else 'NOT_ASSESSED'
            }
        }
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("Check the 'results' folder for saved plots and visualizations.")
        print("=" * 60)
        return complete_results

    def load_learning_data_as_exog(self, learning_data_path):
        """Load and align learning data as exogenous variables for ARIMAX.
        Args:
            learning_data_path (str): Path to persistent learning data
        Returns:
            pd.DataFrame or None: Exogenous variables aligned to main series
        """
        import json
        try:
            with open(learning_data_path, 'r') as f:
                learning_json = json.load(f)
            # Example: Use RSI and volatility from 'stocks' learning data
            stock_entries = learning_json.get('learning_data', {}).get('stocks', [])
            # Build DataFrame indexed by timestamp
            import pandas as pd
            exog_df = pd.DataFrame([
                {
                    'timestamp': entry['timestamp'],
                    'avg_rsi': entry['market_summary'].get('average_rsi', 50),
                    'avg_volatility': entry['market_summary'].get('average_volatility', 0),
                }
                for entry in stock_entries if 'market_summary' in entry
            ])
            exog_df['timestamp'] = pd.to_datetime(exog_df['timestamp'])
            exog_df = exog_df.set_index('timestamp')
            # Make exog_df index timezone-naive
            if hasattr(exog_df.index, 'tz') and exog_df.index.tz is not None:
                exog_df.index = exog_df.index.tz_localize(None)
            # Align to main series index
            main_index = self.preparation_info['original_series'].index
            # Make main_index timezone-naive
            if hasattr(main_index, 'tz') and main_index.tz is not None:
                main_index = main_index.tz_localize(None)
            exog_aligned = exog_df.reindex(main_index, method='nearest', tolerance=pd.Timedelta('1D'))
            print(f"Loaded exogenous variables from learning data: {exog_aligned.shape}")
            return exog_aligned
        except Exception as e:
            print(f"Could not load exogenous variables from learning data: {e}")
            return None


def main():
    """Main function to run ARIMAX analysis for a user-specified stock ticker."""
    # Prompt user for ticker symbol
    ticker = input("Enter the stock ticker symbol (default: RDDT): ").strip().upper()
    if not ticker:
        ticker = 'RDDT'
    print(f"\n[INFO] Using ticker: {ticker}")
    # Prompt user for lookback period
    lookback = input("Enter lookback period (e.g. 2y, 1y, 6mo, 5y) [default: 2y]: ").strip().lower()
    if not lookback:
        lookback = '2y'
    print(f"[INFO] Using lookback period: {lookback}")

        # Option to use smart_ml_persistent_data.json as exogenous variable
    use_smart_ml = input("Use smart_ml_persistent_data.json as exogenous variable? (y/N): ").strip().lower() == 'y'
    # Pass the ticker to ARIMAXForecaster
    forecaster = ARIMAXForecaster(ticker=ticker)
    # Run complete analysis with or without learning data as exogenous variables
    results = forecaster.run_complete_analysis(period=lookback, forecast_steps=60, use_learning_data=use_smart_ml)
    if results:
        print(f"\nAnalysis completed successfully for {results['ticker']}")
        print(f"Best ARIMAX model: {results['best_arima_params']}")
        # Get first forecast value safely
        forecast_series = results['forecast_results']['forecast']
        if len(forecast_series) > 0:
            next_day_prediction = forecast_series.iloc[0]
            print(f"Next day prediction: ${next_day_prediction:.2f}")
        else:
            print("No forecast values available")
    else:
        print("Analysis failed. Please check the data and try again.")

if __name__ == "__main__":
    main()
