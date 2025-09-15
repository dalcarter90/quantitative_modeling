# ARIMA Stock Price Forecasting

This project implements a research-optimized ARIMA (AutoRegressive Integrated Moving Average) model to forecast stock prices for the OPEN stock ticker.

## Features

- **Research-based lookback period**: Optimized 1-year window based on financial time series literature
- Stock data retrieval using Yahoo Finance API
- Data quality assessment and validation
- Structural stability testing
- Time series data preprocessing and visualization
- ARIMA model parameter optimization
- Price forecasting with confidence intervals
- Performance evaluation metrics
- Interactive plots and visualizations

## Research-Based Optimizations

- **1-Year Lookback Period**: Based on academic research showing optimal balance between statistical reliability and structural consistency
- **Data Quality Validation**: Automatic assessment of observation count and data adequacy
- **Stability Testing**: Detection of potential structural breaks in the time series
- **Efficient Parameter Search**: Reduced parameter space optimized for shorter time series
- **Trading Day Awareness**: Calculations based on 252 trading days per year

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main analysis:
   ```bash
   python src/arima_forecast.py
   ```

3. Or use the Jupyter notebook for interactive analysis:
   ```bash
   jupyter notebook notebooks/arima_analysis.ipynb
   ```

## Project Structure

```
arima/
├── src/
│   ├── arima_forecast.py      # Main forecasting script
│   ├── data_loader.py         # Stock data retrieval
│   ├── preprocessor.py        # Data preprocessing utilities
│   └── visualizer.py          # Plotting and visualization
├── notebooks/
│   └── arima_analysis.ipynb   # Interactive analysis notebook
├── data/                      # Data storage (CSV files)
├── results/                   # Output plots and results
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Usage

The main script will:
1. Download OPEN stock data
2. Preprocess the time series
3. Find optimal ARIMA parameters
4. Generate forecasts
5. Evaluate model performance
6. Save results and visualizations

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies
