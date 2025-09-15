#!/usr/bin/env python3
"""
Test script to run all forecasting models for OPEN stock
Demonstrates ARIMA, LSTM, GARCH, and Monte Carlo forecasting
"""

import os
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def test_individual_models():
    """Test each model individually to ensure they work"""
    print("🧪 Testing Individual Models")
    print("=" * 50)
    
    # Test Data Loader
    try:
        from data_loader import StockDataLoader
        loader = StockDataLoader()
        data = loader.fetch_stock_data('OPEN', period='6mo', save_to_csv=False)
        print(f"✅ Data Loader: Loaded {len(data)} days of data")
        current_price = data['Close'].iloc[-1]
        print(f"   Current Price: ${current_price:.2f}")
    except Exception as e:
        print(f"❌ Data Loader failed: {e}")
        return False
    
    # Test ARIMA
    try:
        from arima_forecast import ARIMAForecaster
        print("\n📈 Testing ARIMA Model...")
        arima_model = ARIMAForecaster(ticker='OPEN')
        # We need to load data first
        arima_model.load_data(period='6mo')
        arima_forecast = arima_model.forecast(forecast_days=30)
        if arima_forecast is not None and len(arima_forecast) > 0:
            print(f"✅ ARIMA Model: Generated {len(arima_forecast)} day forecast")
            next_day = arima_forecast.iloc[0] if hasattr(arima_forecast, 'iloc') else arima_forecast[0]
            print(f"   Next day: ${next_day:.2f}")
        else:
            print("⚠️  ARIMA Model: No forecast returned")
    except Exception as e:
        print(f"❌ ARIMA Model failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
    
    # Test LSTM
    try:
        from lstm_model import LSTMForecaster
        print("\n🧠 Testing LSTM Model...")
        lstm_model = LSTMForecaster(sequence_length=30, epochs=50)
        lstm_forecast = lstm_model.forecast(data, forecast_days=30)
        if lstm_forecast is not None and len(lstm_forecast) > 0:
            print(f"✅ LSTM Model: Generated {len(lstm_forecast)} day forecast")
            print(f"   Next day: ${lstm_forecast[0]:.2f}")
        else:
            print("⚠️  LSTM Model: No forecast generated")
    except Exception as e:
        print(f"❌ LSTM Model failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
    
    # Test GARCH
    try:
        from garch_model import GARCHForecaster
        print("\n📊 Testing GARCH Model...")
        garch_model = GARCHForecaster()
        garch_results = garch_model.forecast(data, forecast_days=30)
        if garch_results and 'price_forecast' in garch_results:
            forecast_len = len(garch_results['price_forecast'])
            print(f"✅ GARCH Model: Generated {forecast_len} day forecast")
            print(f"   Next day: ${garch_results['price_forecast'][0]:.2f}")
        else:
            print("⚠️  GARCH Model: No forecast generated")
    except Exception as e:
        print(f"❌ GARCH Model failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
    
    # Test Monte Carlo
    try:
        from monte_carlo import MonteCarloForecaster
        print("\n🎲 Testing Monte Carlo Models...")
        mc_model = MonteCarloForecaster()
        
        # Test GBM
        gbm_results = mc_model.geometric_brownian_motion(data, forecast_days=30, num_simulations=1000)
        if gbm_results and 'summary_stats' in gbm_results:
            mean_path = gbm_results['summary_stats']['mean_path']
            print(f"✅ Monte Carlo GBM: Generated forecast")
            print(f"   Next day: ${mean_path[1]:.2f}")  # [0] is current price
        else:
            print("⚠️  Monte Carlo GBM: No forecast generated")
        
        # Test Jump Diffusion
        jump_results = mc_model.jump_diffusion(data, forecast_days=30, num_simulations=1000)
        if jump_results and 'summary_stats' in jump_results:
            print("✅ Monte Carlo Jump Diffusion: Generated forecast")
        else:
            print("⚠️  Monte Carlo Jump Diffusion: No forecast generated")
        
    except Exception as e:
        print(f"❌ Monte Carlo Models failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
    
    return True

def test_multi_model():
    """Test the unified multi-model system"""
    print("\n🚀 Testing Multi-Model System")
    print("=" * 50)
    
    try:
        from multi_model_forecast import MultiModelForecaster
        
        # Initialize multi-model forecaster
        multi_forecaster = MultiModelForecaster(
            symbol='OPEN',
            period='6mo',
            forecast_days=30
        )
        
        print("✅ Multi-Model Forecaster initialized")
        
        # Run comparison
        print("\n📊 Running model comparison...")
        comparison_results = multi_forecaster.compare_models()
        
        if comparison_results:
            print("✅ Model comparison completed")
            
            # Display results
            print("\n📈 Forecast Summary:")
            print("-" * 40)
            
            for model_name, results in comparison_results.items():
                if results and 'forecast' in results:
                    forecast = results['forecast']
                    if forecast is not None and len(forecast) > 0:
                        next_day = forecast[0] if hasattr(forecast, '__getitem__') else forecast.iloc[0]
                        print(f"{model_name:12s}: ${next_day:.2f}")
                    else:
                        print(f"{model_name:12s}: No forecast")
                else:
                    print(f"{model_name:12s}: Failed")
            
            print("-" * 40)
            
        else:
            print("⚠️  Model comparison returned no results")
        
    except Exception as e:
        print(f"❌ Multi-Model System failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")

def main():
    """Main test function"""
    print("🎯 OPEN Stock Multi-Model Forecasting Test")
    print("=" * 60)
    
    # Test individual models first
    test_individual_models()
    
    # Test multi-model system
    test_multi_model()
    
    print("\n🏁 Testing Complete!")
    print("=" * 60)
    
    # Check for result files
    results_dir = 'results'
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"\n📁 Results folder contains {len(files)} files:")
        for file in sorted(files):
            if file.endswith(('.png', '.html', '.csv', '.txt')):
                print(f"   • {file}")

if __name__ == "__main__":
    main()
