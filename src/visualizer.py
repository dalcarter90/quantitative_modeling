"""
Visualization module for time series analysis and ARIMA results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os


class TimeSeriesVisualizer:
    """Class to handle visualization of time series data and ARIMA results."""
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8', results_dir="results"):
        """Initialize the visualizer.
        
        Args:
            figsize (tuple): Default figure size
            style (str): Matplotlib style
            results_dir (str): Directory to save plots
        """
        self.figsize = figsize
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set color palette
        self.colors = sns.color_palette("husl", 8)
    
    def plot_price_series(self, data, price_column='Close', ticker='Stock', save=True):
        """Plot stock price time series.
        
        Args:
            data (pd.DataFrame or pd.Series): Stock data
            price_column (str): Column to plot if DataFrame
            ticker (str): Stock ticker for title
            save (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if isinstance(data, pd.DataFrame):
            series = data[price_column]
            title = f"{ticker} {price_column} Price Over Time"
        else:
            series = data
            title = f"{ticker} Price Over Time"
        
        ax.plot(series.index, series.values, linewidth=1.5, color=self.colors[0])
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filename = f"{ticker.lower()}_price_series.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        return fig
    
    def plot_stationarity_test(self, original_series, transformed_series, ticker='Stock', save=True):
        """Plot original vs transformed series for stationarity comparison.
        
        Args:
            original_series (pd.Series): Original time series
            transformed_series (pd.Series): Transformed (stationary) series
            ticker (str): Stock ticker for title
            save (bool): Whether to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] + 2))
        
        # Original series
        axes[0].plot(original_series.index, original_series.values, 
                    linewidth=1.5, color=self.colors[0])
        axes[0].set_title(f"{ticker} - Original Price Series", fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Transformed series
        axes[1].plot(transformed_series.index, transformed_series.values, 
                    linewidth=1.5, color=self.colors[1])
        axes[1].set_title(f"{ticker} - Transformed Series (Stationary)", fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Transformed Values', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = f"{ticker.lower()}_stationarity_comparison.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        return fig
    
    def plot_acf_pacf(self, series, lags=40, ticker='Stock', save=True):
        """Plot ACF and PACF for ARIMA parameter estimation.
        
        Args:
            series (pd.Series): Time series
            lags (int): Number of lags to plot
            ticker (str): Stock ticker for title
            save (bool): Whether to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] + 2))
        
        # ACF plot
        plot_acf(series.dropna(), ax=axes[0], lags=lags, alpha=0.05)
        axes[0].set_title(f"{ticker} - Autocorrelation Function (ACF)", fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # PACF plot
        plot_pacf(series.dropna(), ax=axes[1], lags=lags, alpha=0.05)
        axes[1].set_title(f"{ticker} - Partial Autocorrelation Function (PACF)", fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{ticker.lower()}_acf_pacf.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        return fig
    
    def plot_decomposition(self, series, period=252, model='additive', ticker='Stock', save=True):
        """Plot seasonal decomposition of time series.
        
        Args:
            series (pd.Series): Time series to decompose
            period (int): Seasonal period
            model (str): 'additive' or 'multiplicative'
            ticker (str): Stock ticker for title
            save (bool): Whether to save the plot
        """
        try:
            decomposition = seasonal_decompose(series, model=model, period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=(self.figsize[0], self.figsize[1] + 4))
            
            # Original
            axes[0].plot(decomposition.observed.index, decomposition.observed.values, 
                        linewidth=1.5, color=self.colors[0])
            axes[0].set_title(f"{ticker} - Original Series", fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomposition.trend.index, decomposition.trend.values, 
                        linewidth=1.5, color=self.colors[1])
            axes[1].set_title("Trend Component", fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 
                        linewidth=1.5, color=self.colors[2])
            axes[2].set_title("Seasonal Component", fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].plot(decomposition.resid.index, decomposition.resid.values, 
                        linewidth=1.5, color=self.colors[3])
            axes[3].set_title("Residual Component", fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Date', fontsize=12)
            axes[3].grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in axes:
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save:
                filename = f"{ticker.lower()}_decomposition.png"
                filepath = os.path.join(self.results_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {filepath}")
            
            plt.show()
            return fig
            
        except Exception as e:
            print(f"Error in decomposition plot: {str(e)}")
            return None
    
    def plot_forecast(self, historical_data, forecast, confidence_intervals=None, 
                     ticker='Stock', forecast_periods=30, save=True):
        """Plot ARIMA forecast results.
        Args:
            historical_data (pd.Series): Historical price data
            forecast (pd.Series or np.array): Forecasted values
            confidence_intervals (tuple): Lower and upper confidence intervals
            ticker (str): Stock ticker for title
            forecast_periods (int): Number of periods forecasted
            save (bool): Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot historical data (last 252 trading days for context)
        hist_context = historical_data.tail(252) if len(historical_data) > 252 else historical_data
        ax.plot(hist_context.index, hist_context.values, 
                linewidth=1.5, color=self.colors[0], label='Historical Prices')

        # Create forecast dates
        last_date = historical_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                      periods=len(forecast), freq='D')

        # Plot forecast
        ax.plot(forecast_dates, forecast, 
                linewidth=2, color=self.colors[1], label='Forecast', linestyle='--')

        # Plot confidence intervals if provided
        if confidence_intervals is not None:
            lower_ci, upper_ci = confidence_intervals
            # Ensure lower_ci and upper_ci are numeric numpy arrays
            lower_ci = np.asarray(lower_ci, dtype=float)
            upper_ci = np.asarray(upper_ci, dtype=float)
            ax.fill_between(forecast_dates, lower_ci, upper_ci, 
                            color=self.colors[1], alpha=0.3, label='Confidence Interval')

        # --- Add trend line to forecast ---
        # Combine historical and forecast for trend estimation (optional: just forecast)
        all_dates = np.concatenate([hist_context.index.values, forecast_dates.values])
        all_prices = np.concatenate([hist_context.values, forecast])
        x = np.arange(len(all_dates))
        z = np.polyfit(x, all_prices, 1)
        trend = np.poly1d(z)
        ax.plot(all_dates, trend(x), color='orange', linestyle=':', linewidth=2, label='Trend Line')
        # --- End trend line ---

        ax.set_title(f"{ticker} - ARIMA Price Forecast", fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save:
            filename = f"{ticker.lower()}_forecast.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")

        plt.show()
        return fig
    
    def plot_residuals(self, residuals, ticker='Stock', save=True):
        """Plot residual analysis for model diagnostics.
        
        Args:
            residuals (pd.Series or np.array): Model residuals
            ticker (str): Stock ticker for title
            save (bool): Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0] + 2, self.figsize[1]))
        
        # Residuals over time
        axes[0, 0].plot(residuals, linewidth=1, color=self.colors[0])
        axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, color=self.colors[1])
        axes[0, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ACF of residuals
        plot_acf(residuals, ax=axes[1, 1], lags=20, alpha=0.05)
        axes[1, 1].set_title('ACF of Residuals', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{ticker.lower()}_residual_analysis.png"
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        return fig
    
    def create_interactive_forecast(self, historical_data, forecast, confidence_intervals=None, 
                                  ticker='Stock', save=True):
        """Create interactive forecast plot using Plotly.
        
        Args:
            historical_data (pd.Series): Historical price data
            forecast (pd.Series or np.array): Forecasted values
            confidence_intervals (tuple): Lower and upper confidence intervals
            ticker (str): Stock ticker for title
            save (bool): Whether to save the plot
        """
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))
        
        # Create forecast dates
        last_date = historical_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=len(forecast), freq='D')
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence intervals
        if confidence_intervals is not None:
            lower_ci, upper_ci = confidence_intervals
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=upper_ci,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=lower_ci,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                name='Confidence Interval',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f"{ticker} - Interactive ARIMA Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            width=1000,
            height=600
        )
        
        if save:
            filename = f"{ticker.lower()}_interactive_forecast.html"
            filepath = os.path.join(self.results_dir, filename)
            fig.write_html(filepath)
            print(f"Interactive plot saved to {filepath}")
        
        fig.show()
        return fig


if __name__ == "__main__":
    # Example usage
    from data_loader import StockDataLoader
    
    loader = StockDataLoader()
    data = loader.fetch_stock_data("OPEN", period="2y")
    
    if data is not None:
        visualizer = TimeSeriesVisualizer()
        
        # Plot price series
        visualizer.plot_price_series(data, ticker="OPEN")
        
        # Plot decomposition
        visualizer.plot_decomposition(data['Close'], ticker="OPEN")
