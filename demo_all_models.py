#!/usr/bin/env python3
"""
Simple demonstration of all forecasting models for OPEN stock
Shows ARIMA, LSTM, GARCH, and Monte Carlo forecasting in action
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def main():
    print("🚀 OPEN Stock Multi-Model Forecasting Demo")
    print("=" * 60)
    
    # Load OPEN stock data
    from data_loader import StockDataLoader
    
    print("📊 Loading OPEN stock data...")
    loader = StockDataLoader()
    data = loader.fetch_stock_data('OPEN', period='6mo', save_to_csv=False)
    
    current_price = data['Close'].iloc[-1]
    print(f"✅ Loaded {len(data)} days of data")
    print(f"📈 Current OPEN price: ${current_price:.2f}")
    print("\n" + "=" * 60)
    
    forecast_results = {}
    
    # 1. ARIMA Model
    print("\n🔮 1. ARIMA Forecasting")
    print("-" * 30)
    try:
        from arima_forecast import ARIMAForecaster
        
        arima_model = ARIMAForecaster(ticker='OPEN')
        arima_model.load_data(period='6mo')
        arima_model.prepare_data()
        arima_model.find_best_params()
        arima_model.fit_model()
        
        arima_forecast = arima_model.forecast(forecast_days=10)
        if arima_forecast is not None and len(arima_forecast) > 0:
            next_day = arima_forecast.iloc[0] if hasattr(arima_forecast, 'iloc') else arima_forecast[0]
            final_day = arima_forecast.iloc[-1] if hasattr(arima_forecast, 'iloc') else arima_forecast[-1]
            print(f"✅ ARIMA: Next day ${next_day:.2f} → 10-day ${final_day:.2f}")
            forecast_results['ARIMA'] = next_day
        else:
            print("❌ ARIMA: No forecast generated")
    except Exception as e:
        print(f"❌ ARIMA failed: {e}")
    
    # 2. LSTM Model
    print("\n🧠 2. LSTM Neural Network")
    print("-" * 30)
    try:
        from lstm_model import LSTMForecaster
        
        lstm_model = LSTMForecaster(sequence_length=30, epochs=20, batch_size=16)
        print("   Training LSTM model (this may take a moment)...")
        lstm_forecast = lstm_model.forecast(data, forecast_days=10)
        
        if lstm_forecast is not None and len(lstm_forecast) > 0:
            next_day = lstm_forecast[0]
            final_day = lstm_forecast[-1]
            print(f"✅ LSTM: Next day ${next_day:.2f} → 10-day ${final_day:.2f}")
            forecast_results['LSTM'] = next_day
        else:
            print("❌ LSTM: No forecast generated")
    except Exception as e:
        print(f"❌ LSTM failed: {e}")
    
    # 3. GARCH Model
    print("\n📊 3. GARCH Volatility Model")
    print("-" * 30)
    try:
        from garch_model import GARCHForecaster
        
        garch_model = GARCHForecaster()
        garch_results = garch_model.forecast(data, forecast_days=10)
        
        if garch_results and 'price_forecast' in garch_results:
            next_day = garch_results['price_forecast'][0]
            final_day = garch_results['price_forecast'][-1]
            print(f"✅ GARCH: Next day ${next_day:.2f} → 10-day ${final_day:.2f}")
            forecast_results['GARCH'] = next_day
            
            # Show volatility info
            if 'volatility_forecast' in garch_results:
                vol = garch_results['volatility_forecast'][0]
                print(f"   📈 Next day volatility: {vol:.4f}")
        else:
            print("❌ GARCH: No forecast generated")
    except Exception as e:
        print(f"❌ GARCH failed: {e}")
    
    # 4. Monte Carlo Simulations
    print("\n🎲 4. Monte Carlo Simulations")
    print("-" * 30)
    try:
        from monte_carlo import MonteCarloForecaster
        
        mc_model = MonteCarloForecaster()
        
        # Geometric Brownian Motion
        print("   Running GBM simulation...")
        gbm_results = mc_model.geometric_brownian_motion(data, forecast_days=10, num_simulations=5000)
        
        if gbm_results and 'summary_stats' in gbm_results:
            mean_path = gbm_results['summary_stats']['mean_path']
            next_day = mean_path[1]  # [0] is current price
            final_day = mean_path[-1]
            print(f"✅ MC-GBM: Next day ${next_day:.2f} → 10-day ${final_day:.2f}")
            forecast_results['MC-GBM'] = next_day
            
            # Show confidence intervals
            ci_lower = gbm_results['summary_stats']['confidence_intervals']['lower']
            ci_upper = gbm_results['summary_stats']['confidence_intervals']['upper']
            print(f"   📊 95% CI: ${ci_lower[1]:.2f} - ${ci_upper[1]:.2f}")
        else:
            print("❌ MC-GBM: No forecast generated")
        
        # Jump Diffusion Model
        print("   Running Jump Diffusion simulation...")
        jump_results = mc_model.jump_diffusion(data, forecast_days=10, num_simulations=5000)
        
        if jump_results and 'summary_stats' in jump_results:
            mean_path = jump_results['summary_stats']['mean_path']
            next_day = mean_path[1]
            final_day = mean_path[-1]
            print(f"✅ MC-Jump: Next day ${next_day:.2f} → 10-day ${final_day:.2f}")
            forecast_results['MC-Jump'] = next_day
        else:
            print("❌ MC-Jump: No forecast generated")
            
    except Exception as e:
        print(f"❌ Monte Carlo failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FORECAST SUMMARY")
    print("=" * 60)
    print(f"Current Price: ${current_price:.2f}")
    print("-" * 30)
    
    if forecast_results:
        for model, price in forecast_results.items():
            change = ((price - current_price) / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"{model:10s}: ${price:.2f} ({change:+.1f}%) {direction}")
        
        # Calculate ensemble average
        avg_forecast = sum(forecast_results.values()) / len(forecast_results)
        avg_change = ((avg_forecast - current_price) / current_price) * 100
        avg_direction = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
        
        print("-" * 30)
        print(f"{'Ensemble':10s}: ${avg_forecast:.2f} ({avg_change:+.1f}%) {avg_direction}")
    else:
        print("❌ No successful forecasts generated")
    
    print("\n🎯 Multi-model forecasting complete!")

if __name__ == "__main__":
    main()
