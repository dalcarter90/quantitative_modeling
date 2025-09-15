"""
Data loader module for fetching stock data using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


class StockDataLoader:
    """Class to handle stock data retrieval and saving."""
    
    def __init__(self, data_dir="data"):
        """Initialize the data loader.
        
        Args:
            data_dir (str): Directory to save data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_stock_data(self, ticker, period="5y", save_to_csv=True):
        """Fetch stock data for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'OPEN')
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            save_to_csv (bool): Whether to save data to CSV file
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        print(f"Fetching data for {ticker}...")
        
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Add ticker column
            data['Ticker'] = ticker
            
            # Save to CSV if requested
            if save_to_csv:
                filename = f"{ticker}_{period}_data.csv"
                filepath = os.path.join(self.data_dir, filename)
                data.to_csv(filepath)
                print(f"Data saved to {filepath}")
            
            print(f"Successfully fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def load_saved_data(self, ticker, period="5y"):
        """Load previously saved stock data from CSV.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period used when saving
            
        Returns:
            pd.DataFrame: Loaded stock data or None if file not found
        """
        filename = f"{ticker}_{period}_data.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Loaded data from {filepath}")
            return data
        except FileNotFoundError:
            print(f"No saved data found at {filepath}")
            return None
    
    def get_stock_info(self, ticker):
        """Get basic information about the stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'N/A')
            }
        except Exception as e:
            print(f"Error getting stock info: {str(e)}")
            return None


if __name__ == "__main__":
    # Example usage
    loader = StockDataLoader()
    data = loader.fetch_stock_data("OPEN", period="5y")
    if data is not None:
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Columns: {list(data.columns)}")
