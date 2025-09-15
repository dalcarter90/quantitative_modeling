"""
Multi-Model Forecasting System
=============================

This module provides a unified interface for running multiple forecasting models
including ARIMA, LSTM, GARCH, and Monte Carlo simulations.

Author: AI Assistant
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom models
try:
    from .data_loader import StockDataLoader as DataLoader
    from .preprocessor import TimeSeriesPreprocessor as Preprocessor
    from .visualizer import TimeSeriesVisualizer as Visualizer
    from .arima_forecast import ARIMAForecaster
    from .lstm_model import LSTMForecaster
    from .garch_model import GARCHForecaster
    from .monte_carlo import MonteCarloForecaster
    from .technical_indicators import TechnicalIndicators
    from .news_sentiment import NewsSentimentAnalyzer
except ImportError:
    # For direct execution
    from data_loader import StockDataLoader as DataLoader
    from preprocessor import TimeSeriesPreprocessor as Preprocessor
    from visualizer import TimeSeriesVisualizer as Visualizer
    from arima_forecast import ARIMAForecaster
    from lstm_model import LSTMForecaster
    from garch_model import GARCHForecaster
    from monte_carlo import MonteCarloForecaster
    from technical_indicators import TechnicalIndicators
    from news_sentiment import NewsSentimentAnalyzer
    from news_sentiment import NewsSentimentAnalyzer

class MultiModelForecaster:
    """
    Unified forecasting system that combines multiple models.
    
    Attributes:
        symbol (str): Stock symbol
        models (dict): Dictionary of fitted models
        forecasts (dict): Dictionary of forecasting results
        data (pd.DataFrame): Historical price data
        technical_indicators (dict): Technical analysis indicators
        sentiment_data (dict): News sentiment analysis results
    """
    
    def __init__(self, symbol, period='6mo', forecast_days=60):
        """
        Initialize multi-model forecaster.
        
        Args:
            symbol (str): Stock symbol (e.g., 'OPEN')
            period (str): Data period ('6mo', '1y', '2y', etc.)
            forecast_days (int): Number of days to forecast
        """
        self.symbol = symbol.upper()
        self.period = period
        self.forecast_days = forecast_days
        self.models = {}
        self.forecasts = {}
        self.data = None
        self.technical_indicators = {}
        self.sentiment_data = {}
        self.results_summary = {}
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.visualizer = Visualizer()
        self.tech_indicators = TechnicalIndicators()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        
    def load_and_prepare_data(self):
        """Load and prepare data for modeling."""
        print(f"Loading data for {self.symbol}...")
        
        # Load price data
        self.data = self.data_loader.fetch_stock_data(self.symbol, period=self.period, save_to_csv=False)
        
        if self.data is None or len(self.data) < 50:
            raise ValueError(f"Insufficient data for {self.symbol}")
        
        print(f"Loaded {len(self.data)} data points from {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"Available columns: {list(self.data.columns)}")
        
        # Add technical indicators
        print("Calculating technical indicators...")
        self.technical_indicators = self.tech_indicators.enhance_price_series(self.data, price_column='Close')
        
        # Add sentiment analysis
        print("Analyzing market sentiment...")
        try:
            self.sentiment_data = self.sentiment_analyzer.analyze_sentiment(self.symbol, self.data)
        except Exception as e:
            print(f"Warning: Sentiment analysis failed: {e}")
            self.sentiment_data = {}
        
        print("Data preparation complete!")
    
    def fit_arima_model(self):
        """Fit ARIMA model."""
        print("\n" + "="*60)
        print("FITTING ARIMA MODEL")
        print("="*60)
        
        try:
            # Initialize ARIMA model with the correct parameters
            arima_model = ARIMAForecaster(ticker=self.symbol)
            
            # Load and prepare data
            arima_model.load_and_prepare_data(period=self.period)
            
            # Find best parameters and fit model
            arima_model.find_best_arima_params()
            arima_model.fit_model()
            
            # Generate forecasts
            forecast = arima_model.forecast(steps=self.forecast_days)
            
            self.models['ARIMA'] = arima_model
            self.forecasts['ARIMA'] = {
                'forecast': forecast,
                'model': arima_model,
                'method': 'ARIMA'
            }
            
            print("✓ ARIMA model fitted successfully")
            return True
            
        except Exception as e:
            print(f"✗ ARIMA model failed: {e}")
            return False
            
            print(f"✓ ARIMA model fitted successfully")
            print(f"  Model order: {self.forecasts['ARIMA']['model_info']['order']}")
            print(f"  AIC: {self.forecasts['ARIMA']['model_info']['aic']:.2f}")
            
        except Exception as e:
            print(f"✗ ARIMA model failed: {str(e)}")
            self.forecasts['ARIMA'] = None
    
    def fit_lstm_model(self):
        """Fit LSTM model."""
        print("\n" + "="*60)
        print("FITTING LSTM MODEL")
        print("="*60)
        
        try:
            # Initialize LSTM model
            lstm_model = LSTMForecaster(
                sequence_length=60,
                epochs=50,  # Reduced for faster training
                batch_size=32
            )
            
            # Prepare data from the dataframe
            prices = self.data['Close'].values
            X_train, y_train, X_test, y_test, train_data, test_data = lstm_model.prepare_data(
                prices, train_size=0.8
            )
            
            # Train model
            if len(X_train) > 0 and len(X_test) > 0:
                lstm_model.train(X_train, y_train, X_test, y_test)
                
                # Generate forecast
                last_sequence = train_data[-lstm_model.sequence_length:]
                forecast = lstm_model.forecast(last_sequence, steps=self.forecast_days)
                
                # Evaluate model
                metrics = lstm_model.evaluate(X_test, y_test)
                
                # Store results
                self.models['LSTM'] = lstm_model
                self.forecasts['LSTM'] = {
                    'forecast': forecast,
                    'metrics': metrics,
                    'model_info': {
                        'sequence_length': lstm_model.sequence_length,
                        'epochs_trained': len(lstm_model.history.history['loss']) if lstm_model.history else 0
                    }
                }
                
                print(f"✓ LSTM model fitted successfully")
                print(f"  RMSE: {metrics['RMSE']:.4f}")
                print(f"  MAE: {metrics['MAE']:.4f}")
                print(f"  MAPE: {metrics['MAPE']:.2f}%")
            else:
                print("✗ Insufficient data for LSTM training")
                
        except Exception as e:
            print(f"✗ LSTM model failed: {str(e)}")
            self.forecasts['LSTM'] = None
    
    def fit_garch_model(self):
        """Fit GARCH model."""
        print("\n" + "="*60)
        print("FITTING GARCH MODEL")
        print("="*60)
        
        try:
            # Initialize GARCH model
            garch_model = GARCHForecaster(model_type='GARCH', p=1, q=1)
            
            # Prepare returns
            returns = garch_model.prepare_returns(self.data['Close'])
            
            # Test for ARCH effects
            arch_test = garch_model.test_arch_effects()
            
            # Fit model
            garch_model.fit(returns)
            
            # Forecast volatility
            vol_forecast = garch_model.forecast_volatility(horizon=self.forecast_days)
            
            # Generate return forecasts with confidence intervals
            return_forecast = garch_model.forecast_returns(
                vol_forecast['volatility_forecast']
            )
            
            # Calculate VaR
            var_result = garch_model.calculate_var(
                vol_forecast['volatility_forecast'],
                confidence_level=0.05
            )
            
            # Convert to price forecasts (simplified)
            current_price = self.data['Close'].iloc[-1]
            price_forecast = current_price * (1 + return_forecast['mean_forecast'] / 100)
            
            # Store results
            self.models['GARCH'] = garch_model
            self.forecasts['GARCH'] = {
                'volatility_forecast': vol_forecast['volatility_forecast'],
                'return_forecast': return_forecast,
                'price_forecast': price_forecast,
                'var_results': var_result,
                'model_info': {
                    'arch_test_pvalue': arch_test['p_value'],
                    'model_type': garch_model.model_type
                }
            }
            
            print(f"✓ GARCH model fitted successfully")
            print(f"  ARCH test p-value: {arch_test['p_value']:.4f}")
            print(f"  Average forecasted volatility: {np.mean(vol_forecast['volatility_forecast']):.2f}%")
            
        except Exception as e:
            print(f"✗ GARCH model failed: {str(e)}")
            self.forecasts['GARCH'] = None
    
    def fit_monte_carlo_models(self):
        """Fit Monte Carlo models."""
        print("\n" + "="*60)
        print("FITTING MONTE CARLO MODELS")
        print("="*60)
        
        mc_models = ['GBM', 'Jump', 'Heston']
        
        for model_type in mc_models:
            try:
                print(f"\nFitting {model_type} model...")
                
                # Initialize Monte Carlo model
                mc_model = MonteCarloForecaster(
                    model_type=model_type,
                    n_simulations=5000,  # Reduced for faster computation
                    time_horizon=self.forecast_days
                )
                
                # Estimate parameters
                mc_model.estimate_parameters(self.data['Close'])
                
                # Run simulation
                results = mc_model.run_simulation()
                
                # Calculate risk measures
                var_es = mc_model.calculate_var_es()
                
                # Store results
                model_key = f'MC_{model_type}'
                self.models[model_key] = mc_model
                self.forecasts[model_key] = {
                    'mean_forecast': results['summary_stats']['mean_path'][1:],  # Exclude initial price
                    'percentiles': {
                        '5%': results['summary_stats']['lower_5'][1:],
                        '25%': results['summary_stats']['lower_25'][1:],
                        '75%': results['summary_stats']['upper_75'][1:],
                        '95%': results['summary_stats']['upper_95'][1:]
                    },
                    'var_es': var_es,
                    'final_stats': {
                        'mean': results['summary_stats']['final_mean'],
                        'std': results['summary_stats']['final_std'],
                        'prob_positive': results['summary_stats']['prob_positive']
                    }
                }
                
                print(f"✓ {model_type} model fitted successfully")
                print(f"  Final price mean: ${results['summary_stats']['final_mean']:.2f}")
                print(f"  VaR (95%): {var_es['VaR_95']:.2%}")
                
            except Exception as e:
                print(f"✗ {model_type} model failed: {str(e)}")
                self.forecasts[f'MC_{model_type}'] = None
    
    def _prepare_exog_forecast(self):
        """Prepare exogenous variables for ARIMA forecasting."""
        # Simple approach: use last sentiment value for all forecast periods
        if 'sentiment_score' in self.sentiment_data:
            last_sentiment = self.sentiment_data['sentiment_score'].iloc[-1]
            return pd.DataFrame({
                'sentiment': [last_sentiment] * self.forecast_days
            })
        return None
    
    def compare_models(self):
        """Compare forecasting models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        current_price = self.data['Close'].iloc[-1]
        
        # Create comparison table
        comparison_data = []
        
        for model_name, forecast_data in self.forecasts.items():
            if forecast_data is None:
                continue
            
            try:
                if model_name == 'ARIMA':
                    forecast = forecast_data['forecast']
                    next_day = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
                    final_price = forecast.iloc[-1] if hasattr(forecast, 'iloc') else forecast[-1]
                    
                elif model_name == 'LSTM':
                    forecast = forecast_data['forecast']
                    next_day = forecast[0]
                    final_price = forecast[-1]
                    
                elif model_name == 'GARCH':
                    forecast = forecast_data['price_forecast']
                    next_day = forecast[0]
                    final_price = forecast[-1]
                    
                elif model_name.startswith('MC_'):
                    forecast = forecast_data['mean_forecast']
                    next_day = forecast[0]
                    final_price = forecast[-1]
                    
                else:
                    continue
                
                # Calculate returns
                next_day_return = (next_day - current_price) / current_price * 100
                total_return = (final_price - current_price) / current_price * 100
                
                comparison_data.append({
                    'Model': model_name,
                    'Next Day Price': f'${next_day:.2f}',
                    'Next Day Return': f'{next_day_return:+.2f}%',
                    'Final Price': f'${final_price:.2f}',
                    f'{self.forecast_days}-Day Return': f'{total_return:+.2f}%'
                })
                
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                continue
        
        # Create comparison DataFrame
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(f"\nCurrent Price: ${current_price:.2f}")
            print(f"Forecast Horizon: {self.forecast_days} days")
            print("\nModel Comparison:")
            print(comparison_df.to_string(index=False))
            
            # Store summary
            self.results_summary = {
                'current_price': current_price,
                'forecast_horizon': self.forecast_days,
                'comparison_table': comparison_df
            }
        else:
            print("No valid forecasts available for comparison")
    
    def plot_all_forecasts(self, save_path=None):
        """Plot all model forecasts together."""
        print("\nGenerating forecast comparison plot...")
        
        # Create forecast dates
        last_date = self.data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.forecast_days,
            freq='D'
        )
        
        # Set up the plot
        plt.figure(figsize=(16, 10))
        
        # Plot historical data (last 3 months)
        hist_data = self.data['Close'].tail(90)
        plt.plot(hist_data.index, hist_data.values, 
                label='Historical Prices', color='black', linewidth=2)
        
        # Color palette for models
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        color_idx = 0
        
        # Plot forecasts
        for model_name, forecast_data in self.forecasts.items():
            if forecast_data is None:
                continue
            
            try:
                if model_name == 'ARIMA':
                    forecast = forecast_data['forecast']
                    conf_int = forecast_data['conf_int']
                    
                    plt.plot(forecast_dates, forecast, 
                            label=f'{model_name} Forecast', 
                            color=colors[color_idx % len(colors)], 
                            linestyle='--', linewidth=2)
                    
                    # Add confidence interval
                    if conf_int is not None and len(conf_int) == len(forecast_dates):
                        plt.fill_between(forecast_dates, 
                                       conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                                       alpha=0.2, color=colors[color_idx % len(colors)])
                    
                elif model_name == 'LSTM':
                    forecast = forecast_data['forecast']
                    plt.plot(forecast_dates, forecast, 
                            label=f'{model_name} Forecast',
                            color=colors[color_idx % len(colors)], 
                            linestyle='--', linewidth=2)
                    
                elif model_name == 'GARCH':
                    forecast = forecast_data['price_forecast']
                    plt.plot(forecast_dates, forecast, 
                            label=f'{model_name} Forecast',
                            color=colors[color_idx % len(colors)], 
                            linestyle='--', linewidth=2)
                    
                elif model_name.startswith('MC_'):
                    forecast = forecast_data['mean_forecast']
                    percentiles = forecast_data['percentiles']
                    
                    plt.plot(forecast_dates, forecast, 
                            label=f'{model_name} Mean',
                            color=colors[color_idx % len(colors)], 
                            linestyle='--', linewidth=2)
                    
                    # Add confidence bands
                    plt.fill_between(forecast_dates,
                                   percentiles['5%'], percentiles['95%'],
                                   alpha=0.1, color=colors[color_idx % len(colors)])
                
                color_idx += 1
                
            except Exception as e:
                print(f"Error plotting {model_name}: {e}")
                continue
        
        plt.title(f'{self.symbol} - Multi-Model Forecast Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path=None):
        """Generate comprehensive forecasting report."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        report = []
        report.append(f"Multi-Model Forecasting Report for {self.symbol}")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data period: {self.period}")
        report.append(f"Forecast horizon: {self.forecast_days} days")
        report.append(f"Current price: ${self.data['Close'].iloc[-1]:.2f}")
        report.append("")
        
        # Model summary
        report.append("MODEL SUMMARY")
        report.append("-" * 30)
        successful_models = [name for name, data in self.forecasts.items() if data is not None]
        failed_models = [name for name, data in self.forecasts.items() if data is None]
        
        report.append(f"Successfully fitted models: {len(successful_models)}")
        report.append(f"Models: {', '.join(successful_models)}")
        
        if failed_models:
            report.append(f"Failed models: {', '.join(failed_models)}")
        report.append("")
        
        # Forecast comparison
        if hasattr(self, 'results_summary') and 'comparison_table' in self.results_summary:
            report.append("FORECAST COMPARISON")
            report.append("-" * 30)
            report.append(self.results_summary['comparison_table'].to_string(index=False))
            report.append("")
        
        # Technical indicators summary
        if self.technical_indicators:
            report.append("TECHNICAL INDICATORS")
            report.append("-" * 30)
            rsi = self.technical_indicators.get('RSI', {}).get('RSI')
            if rsi is not None:
                current_rsi = rsi.iloc[-1]
                report.append(f"Current RSI: {current_rsi:.2f}")
                if current_rsi > 70:
                    report.append("  Signal: Overbought")
                elif current_rsi < 30:
                    report.append("  Signal: Oversold")
                else:
                    report.append("  Signal: Neutral")
            
            ema_data = self.technical_indicators.get('EMA')
            if ema_data:
                current_price = self.data['Close'].iloc[-1]
                ema_20 = ema_data.get('EMA_20')
                if ema_20 is not None:
                    current_ema = ema_20.iloc[-1]
                    report.append(f"EMA(20): ${current_ema:.2f}")
                    if current_price > current_ema:
                        report.append("  Signal: Above EMA (Bullish)")
                    else:
                        report.append("  Signal: Below EMA (Bearish)")
            report.append("")
        
        # Sentiment analysis
        if self.sentiment_data:
            report.append("SENTIMENT ANALYSIS")
            report.append("-" * 30)
            if 'sentiment_score' in self.sentiment_data:
                avg_sentiment = self.sentiment_data['sentiment_score'].mean()
                report.append(f"Average sentiment score: {avg_sentiment:.3f}")
                if avg_sentiment > 0.1:
                    report.append("  Overall sentiment: Positive")
                elif avg_sentiment < -0.1:
                    report.append("  Overall sentiment: Negative")
                else:
                    report.append("  Overall sentiment: Neutral")
            report.append("")
        
        # Risk assessment
        report.append("RISK ASSESSMENT")
        report.append("-" * 30)
        
        # Get volatility from GARCH if available
        if 'GARCH' in self.forecasts and self.forecasts['GARCH'] is not None:
            avg_vol = np.mean(self.forecasts['GARCH']['volatility_forecast'])
            report.append(f"Forecasted volatility: {avg_vol:.2f}%")
            
            var_results = self.forecasts['GARCH']['var_results']
            report.append(f"1-day VaR (95%): {var_results['var_percentage'][0]:.2f}%")
        
        # Get Monte Carlo risk measures
        mc_vars = []
        for model_name, forecast_data in self.forecasts.items():
            if model_name.startswith('MC_') and forecast_data is not None:
                var_95 = forecast_data['var_es'].get('VaR_95')
                if var_95:
                    mc_vars.append(var_95)
        
        if mc_vars:
            avg_mc_var = np.mean(mc_vars)
            report.append(f"Monte Carlo average VaR (95%): {avg_mc_var:.2%}")
        
        report.append("")
        
        # Recommendations
        report.append("INVESTMENT RECOMMENDATIONS")
        report.append("-" * 30)
        
        # Analyze consensus
        forecasts_1d = []
        forecasts_final = []
        
        for model_name, forecast_data in self.forecasts.items():
            if forecast_data is None:
                continue
            
            try:
                if model_name == 'ARIMA':
                    forecast = forecast_data['forecast']
                    forecasts_1d.append(forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0])
                    forecasts_final.append(forecast.iloc[-1] if hasattr(forecast, 'iloc') else forecast[-1])
                elif model_name == 'LSTM':
                    forecast = forecast_data['forecast']
                    forecasts_1d.append(forecast[0])
                    forecasts_final.append(forecast[-1])
                elif model_name == 'GARCH':
                    forecast = forecast_data['price_forecast']
                    forecasts_1d.append(forecast[0])
                    forecasts_final.append(forecast[-1])
                elif model_name.startswith('MC_'):
                    forecast = forecast_data['mean_forecast']
                    forecasts_1d.append(forecast[0])
                    forecasts_final.append(forecast[-1])
            except:
                continue
        
        if forecasts_1d and forecasts_final:
            current_price = self.data['Close'].iloc[-1]
            
            avg_1d = np.mean(forecasts_1d)
            avg_final = np.mean(forecasts_final)
            
            return_1d = (avg_1d - current_price) / current_price * 100
            return_final = (avg_final - current_price) / current_price * 100
            
            report.append(f"Consensus 1-day return: {return_1d:+.2f}%")
            report.append(f"Consensus {self.forecast_days}-day return: {return_final:+.2f}%")
            
            if return_1d > 1:
                report.append("Short-term outlook: Bullish")
            elif return_1d < -1:
                report.append("Short-term outlook: Bearish")
            else:
                report.append("Short-term outlook: Neutral")
            
            if return_final > 5:
                report.append("Long-term outlook: Bullish")
            elif return_final < -5:
                report.append("Long-term outlook: Bearish")
            else:
                report.append("Long-term outlook: Neutral")
        
        # Join report and print/save
        report_text = "\n".join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {save_path}")
        
        return report_text
    
    def run_complete_analysis(self, save_plots=True, save_report=True):
        """Run complete multi-model analysis."""
        print(f"Starting complete analysis for {self.symbol}")
        print("=" * 80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Fit all models
        self.fit_arima_model()
        self.fit_lstm_model()
        self.fit_garch_model()
        self.fit_monte_carlo_models()
        
        # Compare models
        self.compare_models()
        
        # Generate visualizations
        if save_plots:
            plot_path = f"results/{self.symbol.lower()}_multi_model_forecast.png"
            self.plot_all_forecasts(save_path=plot_path)
        else:
            self.plot_all_forecasts()
        
        # Generate report
        if save_report:
            report_path = f"results/{self.symbol.lower()}_forecast_report.txt"
            self.generate_report(save_path=report_path)
        else:
            self.generate_report()
        
        print("\n" + "="*80)
        print("COMPLETE ANALYSIS FINISHED!")
        print("="*80)

def main():
    """
    Main function to run multi-model analysis.
    """
    # Run analysis for OPEN stock
    forecaster = MultiModelForecaster(
        symbol='OPEN',
        period='6mo',
        forecast_days=60
    )
    
    forecaster.run_complete_analysis()

if __name__ == "__main__":
    main()
