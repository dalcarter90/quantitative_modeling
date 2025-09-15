"""
News sentiment analysis module for stock forecasting enhancement.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Install with: pip install textblob")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class NewsSentimentAnalyzer:
    """Class to analyze news sentiment for stock price forecasting."""
    
    def __init__(self, ticker='OPEN'):
        """Initialize the news sentiment analyzer.
        
        Args:
            ticker (str): Stock ticker symbol
        """
        self.ticker = ticker
        self.sentiment_data = None
        
    def get_company_news_mock(self, days_back=30):
        """
        Generate mock news sentiment data since we don't have API access.
        In a real implementation, this would connect to news APIs.
        
        Args:
            days_back (int): Number of days to look back for news
            
        Returns:
            pd.DataFrame: DataFrame with dates and sentiment scores
        """
        print(f"Generating mock sentiment data for {self.ticker} (last {days_back} days)")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic sentiment patterns
        np.random.seed(42)  # For reproducible results
        
        # Base sentiment around neutral (0.0) with some trend
        base_sentiment = 0.1  # Slightly positive base
        sentiment_scores = []
        
        for i, date in enumerate(dates):
            # Add some trend and noise
            trend = 0.05 * np.sin(i / 10)  # Gradual oscillation
            noise = np.random.normal(0, 0.2)  # Random noise
            daily_sentiment = base_sentiment + trend + noise
            
            # Clip to reasonable sentiment range [-1, 1]
            daily_sentiment = np.clip(daily_sentiment, -1, 1)
            sentiment_scores.append(daily_sentiment)
        
        sentiment_df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'news_volume': np.random.poisson(3, len(dates)),  # Mock news volume
            'sentiment_strength': np.abs(sentiment_scores)  # Absolute sentiment strength
        })
        
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.set_index('date', inplace=True)
        
        return sentiment_df
    
    def get_news_sentiment_yahoo(self, days_back=30):
        """
        Get basic news sentiment using Yahoo Finance (limited).
        This is a simplified approach since Yahoo doesn't provide sentiment scores.
        
        Args:
            days_back (int): Number of days to look back
            
        Returns:
            pd.DataFrame: DataFrame with sentiment proxy data
        """
        if not YFINANCE_AVAILABLE:
            print("yfinance not available, using mock data")
            return self.get_company_news_mock(days_back)
        
        try:
            # Get stock data for volatility-based sentiment proxy
            stock = yf.Ticker(self.ticker)
            
            # Get recent stock data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back + 10)  # Extra days for calculation
            hist_data = stock.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                print(f"No stock data available for {self.ticker}, using mock sentiment")
                return self.get_company_news_mock(days_back)
            
            # Calculate sentiment proxy based on price movements and volume
            hist_data['price_change'] = hist_data['Close'].pct_change()
            hist_data['volume_ma'] = hist_data['Volume'].rolling(window=5).mean()
            hist_data['volume_ratio'] = hist_data['Volume'] / hist_data['volume_ma']
            
            # Create sentiment proxy
            # Positive returns + high volume = positive sentiment
            # Negative returns + high volume = negative sentiment
            sentiment_proxy = []
            for _, row in hist_data.iterrows():
                if pd.isna(row['price_change']) or pd.isna(row['volume_ratio']):
                    sentiment = 0.0
                else:
                    # Combine price change and volume
                    price_sentiment = np.tanh(row['price_change'] * 10)  # Scale and bound
                    volume_weight = min(row['volume_ratio'], 2.0) / 2.0  # Normalize volume impact
                    sentiment = price_sentiment * volume_weight
                
                sentiment_proxy.append(sentiment)
            
            hist_data['sentiment_proxy'] = sentiment_proxy
            
            # Prepare final dataframe
            sentiment_df = pd.DataFrame({
                'sentiment_score': hist_data['sentiment_proxy'].iloc[-days_back:],
                'news_volume': hist_data['volume_ratio'].iloc[-days_back:],
                'sentiment_strength': np.abs(hist_data['sentiment_proxy'].iloc[-days_back:])
            })
            
            print(f"Generated sentiment proxy for {self.ticker} based on price/volume analysis")
            return sentiment_df
            
        except Exception as e:
            print(f"Error getting sentiment from Yahoo Finance: {e}")
            return self.get_company_news_mock(days_back)
    
    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            tuple: (sentiment_score, confidence)
        """
        if not TEXTBLOB_AVAILABLE:
            return 0.0, 0.5
        
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
            confidence = blob.sentiment.subjectivity  # Range: 0 (objective) to 1 (subjective)
            return sentiment, confidence
        except:
            return 0.0, 0.5
    
    def get_sentiment_features(self, days_back=30):
        """
        Get sentiment features for model enhancement.
        
        Args:
            days_back (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Sentiment features
        """
        # Get sentiment data
        sentiment_df = self.get_news_sentiment_yahoo(days_back)
        
        if sentiment_df is None or sentiment_df.empty:
            print("No sentiment data available")
            return None
        
        # Calculate additional features
        sentiment_df['sentiment_ma_3'] = sentiment_df['sentiment_score'].rolling(window=3).mean()
        sentiment_df['sentiment_ma_7'] = sentiment_df['sentiment_score'].rolling(window=7).mean()
        sentiment_df['sentiment_volatility'] = sentiment_df['sentiment_score'].rolling(window=7).std()
        
        # Sentiment momentum (change in sentiment)
        sentiment_df['sentiment_momentum'] = sentiment_df['sentiment_score'].diff()
        
        # Bullish/bearish periods
        sentiment_df['bullish_signal'] = (sentiment_df['sentiment_score'] > 0.2).astype(int)
        sentiment_df['bearish_signal'] = (sentiment_df['sentiment_score'] < -0.2).astype(int)
        
        self.sentiment_data = sentiment_df
        return sentiment_df
    
    def get_current_sentiment_summary(self):
        """
        Get a summary of current sentiment conditions.
        
        Returns:
            dict: Sentiment summary
        """
        if self.sentiment_data is None:
            self.get_sentiment_features()
        
        if self.sentiment_data is None or self.sentiment_data.empty:
            return {'status': 'No sentiment data available'}
        
        recent_data = self.sentiment_data.tail(7)  # Last week
        
        summary = {
            'current_sentiment': recent_data['sentiment_score'].iloc[-1],
            'avg_sentiment_7d': recent_data['sentiment_score'].mean(),
            'sentiment_trend': 'positive' if recent_data['sentiment_momentum'].mean() > 0 else 'negative',
            'sentiment_volatility': recent_data['sentiment_volatility'].iloc[-1] if not pd.isna(recent_data['sentiment_volatility'].iloc[-1]) else 0,
            'bullish_days_7d': recent_data['bullish_signal'].sum(),
            'bearish_days_7d': recent_data['bearish_signal'].sum(),
            'sentiment_strength': recent_data['sentiment_strength'].mean()
        }
        
        return summary
    
    def enhance_price_data_with_sentiment(self, price_data):
        """
        Enhance price data with sentiment features.
        
        Args:
            price_data (pd.DataFrame): Stock price data
            
        Returns:
            pd.DataFrame: Enhanced data with sentiment features
        """
        if self.sentiment_data is None:
            self.get_sentiment_features(days_back=len(price_data) + 10)
        
        if self.sentiment_data is None:
            print("No sentiment data available for enhancement")
            return price_data
        
        # Align sentiment data with price data
        enhanced_data = price_data.copy()
        
        # Merge sentiment features
        for col in ['sentiment_score', 'sentiment_ma_3', 'sentiment_ma_7', 
                   'sentiment_volatility', 'sentiment_momentum', 
                   'bullish_signal', 'bearish_signal']:
            if col in self.sentiment_data.columns:
                enhanced_data[col] = self.sentiment_data[col].reindex(enhanced_data.index, method='ffill')
        
        # Fill any remaining NaN values
        enhanced_data.fillna(method='ffill', inplace=True)
        enhanced_data.fillna(0, inplace=True)
        
        print(f"Enhanced price data with {len([col for col in enhanced_data.columns if 'sentiment' in col or 'bullish' in col or 'bearish' in col])} sentiment features")
        
        return enhanced_data


def demo_sentiment_analysis():
    """Demonstrate sentiment analysis capabilities."""
    print("=== News Sentiment Analysis Demo ===")
    
    analyzer = NewsSentimentAnalyzer('OPEN')
    
    # Get sentiment features
    sentiment_data = analyzer.get_sentiment_features(days_back=30)
    
    if sentiment_data is not None:
        print(f"\nSentiment data shape: {sentiment_data.shape}")
        print(f"Date range: {sentiment_data.index.min()} to {sentiment_data.index.max()}")
        print(f"\nRecent sentiment scores:")
        print(sentiment_data[['sentiment_score', 'sentiment_ma_7', 'bullish_signal']].tail())
        
        # Get summary
        summary = analyzer.get_current_sentiment_summary()
        print(f"\nCurrent Sentiment Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_sentiment_analysis()
