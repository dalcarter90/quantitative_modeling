
import pandas as pd
"""
GARCH Model for Volatility Forecasting
======================================

This module implements GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
models for volatility forecasting and risk analysis.

Author: AI Assistant
Date: August 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from arch.unitroot import ADF
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class GARCHForecaster:
    def get_simulation_parameters(self, annualize=True):
        """
        Returns annualized drift (mu) and volatility (sigma) for simulation, estimated from the fitted GARCH model.
        Args:
            annualize (bool): Whether to annualize the parameters (default: True)
        Returns:
            dict: {'mu': mu_annual, 'sigma': sigma_annual}
        """
        if self.results is None:
            raise ValueError("Model must be fit before extracting simulation parameters.")
        # The mean of the model (drift)
        mu_daily = self.results.params.get('mu', 0.0) if 'mu' in self.results.params else 0.0
        # The mean of the returns if no explicit mu param
        if mu_daily == 0.0 and self.returns is not None:
            mu_daily = np.mean(self.returns)
        # The average conditional volatility (daily)
        sigma_daily = np.mean(self.results.conditional_volatility)
        if annualize:
            mu_annual = mu_daily * 252
            sigma_annual = sigma_daily * np.sqrt(252)
        else:
            mu_annual = mu_daily
            sigma_annual = sigma_daily
        return {'mu': mu_annual, 'sigma': sigma_annual}
    """
    GARCH-based volatility forecasting model.
    
    Attributes:
        model_type (str): Type of GARCH model
        p (int): GARCH lag order
        q (int): ARCH lag order
        model (arch.univariate): Fitted GARCH model
        returns (pd.Series): Log returns data
        fitted_values (pd.Series): Fitted conditional volatility
    """
    
    def __init__(self, model_type='GARCH', p=1, o=0, q=1, exog=None):
        """
        Initialize GARCH forecaster.
        Args:
            model_type (str): Type of model ('GARCH', 'EGARCH', 'GJR-GARCH', 'GARCH-X')
            p (int): GARCH lag order
            o (int): Asymmetry order (for GJR-GARCH)
            q (int): ARCH lag order
            exog (pd.DataFrame or None): Exogenous variables for GARCH-X
        """
        self.model_type = model_type
        self.p = p
        self.o = o
        self.q = q
        self.model = None
        self.fitted_model = None
        self.returns = None
        self.results = None
        self.exog = exog
        
    def prepare_returns(self, prices):
        """
        Calculate daily percentage returns from price series.
        Args:
            prices (pd.Series): Price time series
        Returns:
            pd.Series: Daily percentage returns
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        # Calculate daily percentage returns
        self.returns = prices.pct_change().dropna()
        self.returns.name = 'pct_return'
        # Check for stationarity
        adf_test = ADF(self.returns)
        print(f"ADF Test Results:")
        print(f"ADF Statistic: {adf_test.stat:.4f}")
        print(f"P-value: {adf_test.pvalue:.4f}")
        print(f"Critical Values: {adf_test.critical_values}")
        if adf_test.pvalue <= 0.05:
            print("✓ Returns series is stationary")
        else:
            print("⚠ Returns series may not be stationary")
        return self.returns
    
    def test_arch_effects(self):
        """
        Test for ARCH effects in the returns series.
        
        Returns:
            dict: ARCH-LM test results
        """
        if self.returns is None:
            raise ValueError("Must prepare returns first")
        
        # Simple ARCH-LM test implementation
        returns_squared = self.returns ** 2
        
        # Create lagged values
        lags = 5
        arch_data = pd.DataFrame({'returns_sq': returns_squared})
        
        for i in range(1, lags + 1):
            arch_data[f'lag_{i}'] = returns_squared.shift(i)
        arch_data = arch_data.fillna(method='ffill').fillna(method='bfill')
        
        # Simple regression R-squared test
        from sklearn.linear_model import LinearRegression
        
        X = arch_data.iloc[:, 1:]  # Lagged values
        y = arch_data.iloc[:, 0]   # Current squared returns
        
        reg = LinearRegression().fit(X, y)
        r_squared = reg.score(X, y)
        
        # LM statistic
        n = len(arch_data)
        lm_stat = n * r_squared
        p_value = 1 - stats.chi2.cdf(lm_stat, lags)
        
        print(f"ARCH-LM Test Results:")
        print(f"LM Statistic: {lm_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value <= 0.05:
            print("✓ ARCH effects detected - GARCH modeling appropriate")
        else:
            print("⚠ No significant ARCH effects detected")
            
        return {
            'lm_statistic': lm_stat,
            'p_value': p_value,
            'r_squared': r_squared
        }
    
    def fit(self, returns=None, exog=None):
        """
        Fit GARCH model to returns data.
        
        Args:
            returns (pd.Series, optional): Returns series to fit
        """
        if returns is not None:
            self.returns = returns
        if exog is not None:
            self.exog = exog
        if self.returns is None:
            raise ValueError("No returns data available")
        # Create GARCH or GARCH-X model
        if self.model_type.upper() == 'GARCH-X':
            if self.exog is None:
                raise ValueError("Exogenous variables (exog) required for GARCH-X")
            self.model = arch_model(
                self.returns,
                vol='Garch',
                p=self.p,
                q=self.q,
                power=2.0,
                mean='Constant',
                dist='normal',
                x=self.exog
            )
        elif self.model_type.upper() in ['GARCH', 'EGARCH', 'GJR-GARCH']:
            # GJR-GARCH is a special case of GARCH with the 'o' parameter
            vol_model = 'Garch' if self.model_type.upper() in ['GARCH', 'GJR-GARCH'] else self.model_type.capitalize()
            # Use self.o if provided (for GJR-GARCH, o=1)
            arch_kwargs = dict(
                vol=vol_model,
                p=self.p,
                q=self.q,
                power=2.0,
                mean='Constant',
                dist='normal'
            )
            if self.o > 0:
                arch_kwargs['o'] = self.o
            self.model = arch_model(
                self.returns,
                **arch_kwargs
            )
        else:
            raise ValueError("Model type must be 'GARCH', 'EGARCH', 'GJR-GARCH', or 'GARCH-X'")
        print(f"Fitting {self.model_type}({self.p},{self.q}) model...")
        self.results = self.model.fit(disp='off')
        self.fitted_values = self.results.conditional_volatility
        print("Model fitted successfully!")
        print(self.results.summary())
        
    def forecast_volatility(self, horizon=30, method='simulation', simulations=1000):
        """
        Forecast volatility using the fitted GARCH model.
        
        Args:
            horizon (int): Forecast horizon
            method (str): Forecasting method ('analytical' or 'simulation')
            simulations (int): Number of simulations if using simulation method
            
        Returns:
            dict: Forecasting results
        """
        if self.results is None:
            raise ValueError("Must fit model before forecasting")
        
        if method == 'analytical':
            # Analytical forecast
            forecast = self.results.forecast(horizon=horizon, method='analytic')
            volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
            
        elif method == 'simulation':
            # Simulation-based forecast
            forecast = self.results.forecast(
                horizon=horizon, 
                method='simulation',
                simulations=simulations
            )
            volatility_forecast = np.sqrt(forecast.variance.mean(axis=0).values)
            
        else:
            raise ValueError("Method must be 'analytical' or 'simulation'")
        
        return {
            'volatility_forecast': volatility_forecast,
            'horizon': horizon,
            'method': method,
            'forecast_object': forecast
        }
    
    def forecast_returns(self, volatility_forecast, confidence_level=0.95):
        """
        Generate return forecasts with confidence intervals.
        
        Args:
            volatility_forecast (np.array): Forecasted volatility
            confidence_level (float): Confidence level for intervals
            
        Returns:
            dict: Return forecasts with confidence intervals
        """
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        # Assume mean return is approximately zero for short-term forecasts
        mean_return = 0
        
        # Calculate confidence intervals
        upper_bound = mean_return + z_score * volatility_forecast
        lower_bound = mean_return - z_score * volatility_forecast
        
        return {
            'mean_forecast': np.full_like(volatility_forecast, mean_return),
            'volatility_forecast': volatility_forecast,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'confidence_level': confidence_level
        }
    
    def backtest_volatility(self, test_size=30):
        """
        Backtest volatility forecasts.
        
        Args:
            test_size (int): Number of periods for backtesting
            
        Returns:
            dict: Backtesting results
        """
        if self.results is None:
            raise ValueError("Must fit model before backtesting")
        
        # Split data
        train_returns = self.returns[:-test_size]
        test_returns = self.returns[-test_size:]
        
        # Refit model on training data
        train_model = arch_model(train_returns, vol='Garch', p=self.p, q=self.q)
        train_results = train_model.fit(disp='off')
        
        # Generate rolling forecasts
        forecasts = []
        actuals = []
        
        for i in range(test_size):
            # One-step ahead forecast
            forecast = train_results.forecast(horizon=1, method='analytic')
            vol_forecast = np.sqrt(forecast.variance.values[-1, 0])
            forecasts.append(vol_forecast)
            
            # Actual volatility (absolute return as proxy)
            actual_vol = abs(test_returns.iloc[i])
            actuals.append(actual_vol)
            
            # Update model with new observation (simplified)
            if i < test_size - 1:
                new_data = self.returns[:-(test_size-i-1)]
                train_model = arch_model(new_data, vol='Garch', p=self.p, q=self.q)
                train_results = train_model.fit(disp='off')
        
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, forecasts)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, forecasts)
        
        return {
            'forecasts': forecasts,
            'actuals': actuals,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    def calculate_var(self, volatility_forecast, confidence_level=0.05):
        """
        Calculate Value at Risk (VaR) and Expected Shortfall (ES) as percentages using GARCH volatility forecasts.
        
        Args:
            volatility_forecast (np.array): Forecasted volatility (in percent)
            confidence_level (float): VaR confidence level (e.g., 0.05 for 95% VaR)
        Returns:
            dict: VaR and ES as percentages
        """
        # Calculate VaR (as percent)
        z_score = stats.norm.ppf(confidence_level)
        var_percentage = z_score * volatility_forecast
        # Calculate Expected Shortfall (Conditional VaR, as percent)
        es_multiplier = stats.norm.pdf(z_score) / confidence_level
        es_percentage = es_multiplier * volatility_forecast
        return {
            'var_percentage': var_percentage,
            'expected_shortfall_percentage': es_percentage,
            'confidence_level': 1 - confidence_level
        }
    
    def plot_volatility(self, forecast_result=None, title="GARCH Volatility Analysis"):
        """
        Plot historical and forecasted volatility.
        
        Args:
            forecast_result (dict, optional): Forecasting results
            title (str): Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Returns
        axes[0, 0].plot(self.returns.index, self.returns.values)
        axes[0, 0].set_title('Log Returns')
        axes[0, 0].set_ylabel('Returns (%)')
        axes[0, 0].grid(True)
        
        # Plot 2: Conditional Volatility
        if self.fitted_values is not None:
            axes[0, 1].plot(self.fitted_values.index, self.fitted_values.values)
            axes[0, 1].set_title('Conditional Volatility')
            axes[0, 1].set_ylabel('Volatility (%)')
            axes[0, 1].grid(True)
        
        # Plot 3: Squared Returns vs Fitted Volatility
        if self.fitted_values is not None:
            squared_returns = self.returns ** 2
            axes[1, 0].plot(squared_returns.index, squared_returns.values, 
                           alpha=0.7, label='Squared Returns')
            axes[1, 0].plot(self.fitted_values.index, self.fitted_values.values ** 2, 
                           color='red', label='Fitted Variance')
            axes[1, 0].set_title('Squared Returns vs Fitted Variance')
            axes[1, 0].set_ylabel('Variance')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot 4: Volatility Forecast
        if forecast_result is not None:
            vol_forecast = forecast_result['volatility_forecast']
            horizon = forecast_result['horizon']
            
            # Create forecast dates
            last_date = self.returns.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
            
            # Plot historical volatility
            axes[1, 1].plot(self.fitted_values.index[-60:], 
                           self.fitted_values.values[-60:], 
                           label='Historical Volatility', color='blue')
            
            # Plot forecast
            axes[1, 1].plot(forecast_dates, vol_forecast, 
                           label='Volatility Forecast', color='red', linestyle='--')
            
            axes[1, 1].set_title('Volatility Forecast')
            axes[1, 1].set_ylabel('Volatility (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_residual_analysis(self):
        """Plot residual analysis for model diagnostics."""
        if self.results is None:
            raise ValueError("Must fit model before residual analysis")
        
        # Get standardized residuals
        residuals = self.results.resid
        std_residuals = residuals / self.fitted_values
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Residuals
        axes[0, 0].plot(residuals.index, residuals.values)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True)
        
        # Plot 2: Standardized Residuals
        axes[0, 1].plot(std_residuals.index, std_residuals.values)
        axes[0, 1].set_title('Standardized Residuals')
        axes[0, 1].set_ylabel('Std. Residuals')
        axes[0, 1].grid(True)
        
        # Plot 3: Q-Q Plot
        stats.probplot(std_residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Standardized Residuals')
        
        # Plot 4: ACF of Squared Residuals
        from statsmodels.tsa.stattools import acf
        
        squared_residuals = std_residuals ** 2
        lags = min(20, len(squared_residuals) // 4)
        acf_values = acf(squared_residuals, nlags=lags)
        
        axes[1, 1].bar(range(len(acf_values)), acf_values)
        axes[1, 1].set_title('ACF of Squared Standardized Residuals')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('ACF')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate GARCH modeling with RDDT stock.
    """
    print("GARCH-X Model Demo")
    print("===================")
    # Prompt user for ticker symbol
    ticker = input("Enter the stock ticker symbol (default: RDDT): ").strip().upper()
    if not ticker:
        ticker = 'RDDT'
    print(f"\n[INFO] Using ticker: {ticker}")
    try:
        from data_loader import StockDataLoader
        data_loader = StockDataLoader()
        stock_data = data_loader.fetch_stock_data(ticker, period="2y")
        if stock_data is None or len(stock_data) < 50:
            raise ValueError(f"Insufficient {ticker} stock data")
        print(f"Loaded {len(stock_data)} days of {ticker} stock data")
        print(f"Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
        close_prices = stock_data['Close']
        # --- Option to use exogenous variable from ml_forecasts.csv ---
        use_ml_csv = input("Use ml_forecasts.csv (ml_vol_forecast) as exogenous variable? (y/N): ").strip().lower() == 'y'
        exog = None
        if use_ml_csv:
            try:
                ml_df = pd.read_csv('ml_forecasts.csv', parse_dates=['date'])
                ml_df = ml_df.set_index('date')
                exog = ml_df.reindex(close_prices.index)[['ml_vol_forecast']].fillna(method='ffill').fillna(0)
                print(f"Loaded exogenous variable from ml_forecasts.csv: {exog.shape}")
            except Exception as e:
                print(f"Could not load exogenous variable from ml_forecasts.csv: {e}")
                exog = pd.DataFrame({'lagged_return': close_prices.pct_change().shift(1)}).fillna(0)
                print("Falling back to lagged returns as exogenous variable.")
        else:
            use_smart_ml = input("Use smart_ml_persistent_data.json as exogenous variable? (y/N): ").strip().lower() == 'y'
            if use_smart_ml:
                import json
                try:
                    with open('smart_ml_persistent_data.json', 'r') as f:
                        smart_ml = json.load(f)
                    stock_entries = smart_ml.get('learning_data', {}).get('stocks', [])
                    exog_df = pd.DataFrame([
                        {
                            'timestamp': entry['timestamp'],
                            'avg_rsi': entry['market_summary'].get('average_rsi', 50),
                            'avg_volatility': entry['market_summary'].get('average_volatility', 0),
                        }
                        for entry in stock_entries if 'market_summary' in entry
                    ])
                    exog_df['timestamp'] = pd.to_datetime(exog_df['timestamp'])
                    exog_df['timestamp'] = exog_df['timestamp'].dt.tz_localize(None)
                    exog_df = exog_df.set_index('timestamp')
                    close_index_naive = close_prices.index.tz_localize(None) if hasattr(close_prices.index, 'tz') and close_prices.index.tz is not None else close_prices.index
                    exog = exog_df.reindex(close_index_naive, method='nearest', tolerance=pd.Timedelta('1D')).fillna(0)
                    print(f"Loaded exogenous variables from smart_ml_persistent_data.json: {exog.shape}")
                except Exception as e:
                    print(f"Could not load exogenous variables from smart_ml_persistent_data.json: {e}")
                    exog = pd.DataFrame({'lagged_return': close_prices.pct_change().shift(1)}).fillna(0)
                    print("Falling back to lagged returns as exogenous variable.")
            else:
                exog = pd.DataFrame({'lagged_return': close_prices.pct_change().shift(1)}).fillna(0)
                print("Using lagged returns as exogenous variable.")

        # --- Instantiate GARCH forecaster ---
        if exog is not None:
           garch_model = GARCHForecaster(model_type='GARCH-X', p=1, q=1, exog=exog)
        else:
           # Activate GJR-GARCH by changing the model_type here
           garch_model = GARCHForecaster(model_type='GJR-GARCH', p=1, q=1)

        # --- CRITICAL STEP: Prepare returns from raw prices ---
        garch_model.prepare_returns(close_prices)

        # --- Test for ARCH effects ---
        print(f"Testing for ARCH effects in {ticker} stock...")
        arch_test = garch_model.test_arch_effects()

        # --- Fit the model using the correctly prepared internal data ---
        print(f"\nFitting GARCH model to {ticker} stock...")
        garch_model.fit()

        # --- Forecast volatility ---
        print(f"\nForecasting {ticker} volatility...")
        forecast_result = garch_model.forecast_volatility(horizon=30)
        print(f"30-day {ticker} volatility forecast: {forecast_result['volatility_forecast'][:5]}...")

        # --- Calculate VaR for ticker ---
        var_result = garch_model.calculate_var(
            forecast_result['volatility_forecast'],
            confidence_level=0.05
        )
        print(f"\n{ticker} 1-day VaR (95%): {var_result['var_percentage'][0]:.2f}%")
        print(f"{ticker} 1-day Expected Shortfall: {var_result['expected_shortfall_percentage'][0]:.2f}%")
    except Exception as e:
        print(f"Error loading {ticker} data: {e}")
        print("Falling back to simulated data...")
        np.random.seed(42)
        n = 500
        prices = np.cumprod(1 + np.random.randn(n) * 0.01) * 100
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        price_series = pd.Series(prices, index=dates)
        exog = pd.DataFrame({'lagged_return': price_series.pct_change().shift(1)}).fillna(0)
        garch_model = GARCHForecaster(model_type='GARCH-X', p=1, q=1, exog=exog)
        garch_model.prepare_returns(price_series)
        print("Testing for ARCH effects...")
        arch_test = garch_model.test_arch_effects()
        print("\nFitting GARCH-X model...")
        garch_model.fit()
        print("\nForecasting volatility...")
        forecast_result = garch_model.forecast_volatility(horizon=30)
        print(f"30-day volatility forecast: {forecast_result['volatility_forecast'][:5]}...")
        var_result = garch_model.calculate_var(
            forecast_result['volatility_forecast'],
            confidence_level=0.05
        )
        print(f"\n1-day VaR (95%): {var_result['var_percentage'][0]:.2f}%")
        print(f"1-day Expected Shortfall: {var_result['expected_shortfall_percentage'][0]:.2f}%")

if __name__ == "__main__":
    main()
