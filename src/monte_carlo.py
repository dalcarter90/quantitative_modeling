"""
Monte Carlo Simulation for Stock Price Forecasting
==================================================

This module implements Monte Carlo simulation methods for stock price forecasting,
including Geometric Brownian Motion, Jump Diffusion, and Stochastic Volatility models.

Author: AI Assistant
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class MonteCarloForecaster:
    """
    Monte Carlo simulation-based forecasting model.
    
    Attributes:
        model_type (str): Type of Monte Carlo model
        n_simulations (int): Number of simulation paths
        time_horizon (int): Forecast time horizon
        dt (float): Time step size
        random_seed (int): Random seed for reproducibility
    """
    
    def __init__(self, model_type='GBM', n_simulations=10000, time_horizon=30, dt=1/252, random_seed=42):
        """
        Initialize Monte Carlo forecaster.
        
        Args:
            model_type (str): Type of model ('GBM', 'Jump', 'Heston')
            n_simulations (int): Number of simulation paths
            time_horizon (int): Forecast horizon in days
            dt (float): Time step (1/252 for daily data)
            random_seed (int): Random seed
        """
        self.model_type = model_type
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.dt = dt
        self.random_seed = random_seed
        self.parameters = {}
        self.simulations = None
        self.summary_stats = None
        
        np.random.seed(self.random_seed)
    
    def estimate_parameters(self, prices):
        """
        Estimate model parameters from historical price data.
        
        Args:
            prices (pd.Series): Historical price series
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        
        # Calculate returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Basic parameters for all models
        self.parameters['mu'] = returns.mean() * 252  # Annualized drift
        self.parameters['sigma'] = returns.std() * np.sqrt(252)  # Annualized volatility
        self.parameters['S0'] = prices.iloc[-1]  # Current price
        
        print(f"Estimated Parameters:")
        print(f"Current Price (S0): ${self.parameters['S0']:.2f}")
        print(f"Annualized Drift (μ): {self.parameters['mu']:.4f}")
        print(f"Annualized Volatility (σ): {self.parameters['sigma']:.4f}")
        
        # Model-specific parameters
        if self.model_type == 'Jump':
            self._estimate_jump_parameters(returns)
        elif self.model_type == 'Heston':
            self._estimate_heston_parameters(returns)
    
    def _estimate_jump_parameters(self, returns):
        """Estimate jump diffusion parameters."""
        # Simple jump detection: outliers beyond 3 standard deviations
        threshold = 3 * returns.std()
        jumps = returns[abs(returns) > threshold]
        
        # Jump intensity (jumps per year)
        self.parameters['lambda_j'] = len(jumps) / len(returns) * 252
        
        # Jump size distribution
        if len(jumps) > 0:
            self.parameters['mu_j'] = jumps.mean()
            self.parameters['sigma_j'] = jumps.std()
        else:
            self.parameters['mu_j'] = 0
            self.parameters['sigma_j'] = 0.01
        
        print(f"Jump Intensity (λ): {self.parameters['lambda_j']:.4f}")
        print(f"Jump Mean (μ_j): {self.parameters['mu_j']:.4f}")
        print(f"Jump Std (σ_j): {self.parameters['sigma_j']:.4f}")
    
    def _estimate_heston_parameters(self, returns):
        """Estimate Heston model parameters (simplified)."""
        # Simplified parameter estimation
        variance = returns.var()
        
        self.parameters['v0'] = variance  # Initial variance
        self.parameters['theta'] = variance  # Long-term variance
        self.parameters['kappa'] = 2.0  # Mean reversion speed
        self.parameters['xi'] = 0.1  # Volatility of volatility
        self.parameters['rho'] = -0.7  # Correlation between price and volatility
        
        print(f"Initial Variance (v0): {self.parameters['v0']:.6f}")
        print(f"Long-term Variance (θ): {self.parameters['theta']:.6f}")
        print(f"Mean Reversion (κ): {self.parameters['kappa']:.4f}")
        print(f"Vol of Vol (ξ): {self.parameters['xi']:.4f}")
        print(f"Correlation (ρ): {self.parameters['rho']:.4f}")
    
    def simulate_gbm(self):
        """
        Simulate stock prices using Geometric Brownian Motion.
        
        Returns:
            np.array: Simulated price paths
        """
        # Parameters
        S0 = self.parameters['S0']
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        
        # Time steps
        t = np.linspace(0, self.time_horizon * self.dt, self.time_horizon + 1)
        
        # Generate random numbers
        np.random.seed(self.random_seed)
        dW = np.random.normal(0, np.sqrt(self.dt), (self.n_simulations, self.time_horizon))
        
        # Initialize price array
        S = np.zeros((self.n_simulations, self.time_horizon + 1))
        S[:, 0] = S0
        
        # Simulate paths
        for i in range(self.time_horizon):
            S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * self.dt + sigma * dW[:, i])
        
        return S
    
    def simulate_jump_diffusion(self):
        """
        Simulate stock prices using Jump Diffusion model.
        
        Returns:
            np.array: Simulated price paths
        """
        # Parameters
        S0 = self.parameters['S0']
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        lambda_j = self.parameters['lambda_j']
        mu_j = self.parameters['mu_j']
        sigma_j = self.parameters['sigma_j']
        
        # Generate random numbers
        np.random.seed(self.random_seed)
        dW = np.random.normal(0, np.sqrt(self.dt), (self.n_simulations, self.time_horizon))
        
        # Jump components
        jump_times = np.random.poisson(lambda_j * self.dt, (self.n_simulations, self.time_horizon))
        jump_sizes = np.random.normal(mu_j, sigma_j, (self.n_simulations, self.time_horizon))
        
        # Initialize price array
        S = np.zeros((self.n_simulations, self.time_horizon + 1))
        S[:, 0] = S0
        
        # Simulate paths
        for i in range(self.time_horizon):
            # Diffusion component
            diffusion = (mu - 0.5 * sigma**2) * self.dt + sigma * dW[:, i]
            
            # Jump component
            jump = jump_times[:, i] * jump_sizes[:, i]
            
            # Update prices
            S[:, i + 1] = S[:, i] * np.exp(diffusion + jump)
        
        return S
    
    def simulate_heston(self):
        """
        Simulate stock prices using Heston Stochastic Volatility model.
        
        Returns:
            tuple: (price_paths, volatility_paths)
        """
        # Parameters
        S0 = self.parameters['S0']
        mu = self.parameters['mu']
        v0 = self.parameters['v0']
        theta = self.parameters['theta']
        kappa = self.parameters['kappa']
        xi = self.parameters['xi']
        rho = self.parameters['rho']
        
        # Generate correlated random numbers
        np.random.seed(self.random_seed)
        Z1 = np.random.normal(0, 1, (self.n_simulations, self.time_horizon))
        Z2 = np.random.normal(0, 1, (self.n_simulations, self.time_horizon))
        
        # Create correlated Brownian motions
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        # Initialize arrays
        S = np.zeros((self.n_simulations, self.time_horizon + 1))
        v = np.zeros((self.n_simulations, self.time_horizon + 1))
        
        S[:, 0] = S0
        v[:, 0] = v0
        
        # Simulate paths using Euler scheme
        for i in range(self.time_horizon):
            # Ensure variance stays positive (reflection)
            v_current = np.maximum(v[:, i], 0)
            
            # Variance process
            dv = kappa * (theta - v_current) * self.dt + xi * np.sqrt(v_current * self.dt) * W2[:, i]
            v[:, i + 1] = v_current + dv
            
            # Price process
            dS = mu * S[:, i] * self.dt + np.sqrt(v_current) * S[:, i] * np.sqrt(self.dt) * W1[:, i]
            S[:, i + 1] = S[:, i] + dS
        
        return S, v
    
    def run_simulation(self):
        """
        Run Monte Carlo simulation based on selected model type.
        
        Returns:
            dict: Simulation results
        """
        print(f"Running {self.n_simulations:,} {self.model_type} simulations...")
        
        if self.model_type == 'GBM':
            self.simulations = self.simulate_gbm()
            volatility_paths = None
            
        elif self.model_type == 'Jump':
            self.simulations = self.simulate_jump_diffusion()
            volatility_paths = None
            
        elif self.model_type == 'Heston':
            self.simulations, volatility_paths = self.simulate_heston()
            
        else:
            raise ValueError("Model type must be 'GBM', 'Jump', or 'Heston'")
        
        # Calculate summary statistics
        self._calculate_summary_stats()
        
        return {
            'price_paths': self.simulations,
            'volatility_paths': volatility_paths,
            'summary_stats': self.summary_stats
        }
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics from simulations."""
        # Final prices
        final_prices = self.simulations[:, -1]
        
        # Price statistics at each time step
        mean_path = np.mean(self.simulations, axis=0)
        median_path = np.median(self.simulations, axis=0)
        std_path = np.std(self.simulations, axis=0)
        
        # Confidence intervals
        lower_5 = np.percentile(self.simulations, 5, axis=0)
        lower_25 = np.percentile(self.simulations, 25, axis=0)
        upper_75 = np.percentile(self.simulations, 75, axis=0)
        upper_95 = np.percentile(self.simulations, 95, axis=0)
        
        self.summary_stats = {
            'final_prices': final_prices,
            'mean_path': mean_path,
            'median_path': median_path,
            'std_path': std_path,
            'lower_5': lower_5,
            'lower_25': lower_25,
            'upper_75': upper_75,
            'upper_95': upper_95,
            'final_mean': np.mean(final_prices),
            'final_median': np.median(final_prices),
            'final_std': np.std(final_prices),
            'prob_positive': np.mean(final_prices > self.parameters['S0']),
            'prob_above_threshold': {}
        }
        
        # Calculate probabilities for different thresholds
        S0 = self.parameters['S0']
        thresholds = [0.9 * S0, 1.1 * S0, 1.2 * S0, 1.5 * S0]
        
        for threshold in thresholds:
            prob = np.mean(final_prices > threshold)
            self.summary_stats['prob_above_threshold'][f'{threshold:.2f}'] = prob
    
    def calculate_var_es(self, confidence_levels=[0.95, 0.99]):
        """
        Calculate Value at Risk and Expected Shortfall.
        
        Args:
            confidence_levels (list): Confidence levels for VaR calculation
            
        Returns:
            dict: VaR and ES results
        """
        if self.simulations is None:
            raise ValueError("Must run simulation first")
        
        # Calculate returns
        final_prices = self.simulations[:, -1]
        S0 = self.parameters['S0']
        returns = (final_prices - S0) / S0
        
        var_es_results = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # VaR (negative of percentile for losses)
            var = -np.percentile(returns, alpha * 100)
            
            # Expected Shortfall (average of returns below VaR)
            tail_returns = returns[returns <= -var]
            es = -np.mean(tail_returns) if len(tail_returns) > 0 else var
            
            var_es_results[f'VaR_{int(conf_level*100)}'] = var
            var_es_results[f'ES_{int(conf_level*100)}'] = es
        
        return var_es_results
    
    def option_pricing(self, strike_price, option_type='call', risk_free_rate=0.05):
        """
        Price European options using Monte Carlo simulation.
        
        Args:
            strike_price (float): Option strike price
            option_type (str): 'call' or 'put'
            risk_free_rate (float): Risk-free interest rate
            
        Returns:
            dict: Option pricing results
        """
        if self.simulations is None:
            raise ValueError("Must run simulation first")
        
        # Final stock prices
        final_prices = self.simulations[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - strike_price, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(strike_price - final_prices, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Discount to present value
        time_to_maturity = self.time_horizon * self.dt
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        option_price = discount_factor * np.mean(payoffs)
        
        # Calculate confidence interval for option price
        option_prices = discount_factor * payoffs
        price_std = np.std(option_prices)
        price_se = price_std / np.sqrt(self.n_simulations)
        
        confidence_interval = [
            option_price - 1.96 * price_se,
            option_price + 1.96 * price_se
        ]
        
        return {
            'option_price': option_price,
            'confidence_interval': confidence_interval,
            'standard_error': price_se,
            'payoffs': payoffs
        }
    
    def plot_simulation_results(self, n_paths_to_plot=100):
        """
        Plot Monte Carlo simulation results.
        
        Args:
            n_paths_to_plot (int): Number of individual paths to plot
        """
        if self.simulations is None:
            raise ValueError("Must run simulation first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Time axis
        time_axis = np.arange(self.time_horizon + 1)
        
        # Plot 1: Sample paths with confidence intervals
        # Plot sample paths
        sample_indices = np.random.choice(self.n_simulations, n_paths_to_plot, replace=False)
        for i in sample_indices:
            axes[0, 0].plot(time_axis, self.simulations[i, :], alpha=0.1, color='blue')
        
        # Plot mean and confidence intervals
        axes[0, 0].plot(time_axis, self.summary_stats['mean_path'], 
                       color='red', linewidth=2, label='Mean')
        axes[0, 0].fill_between(time_axis, 
                               self.summary_stats['lower_5'], 
                               self.summary_stats['upper_95'],
                               alpha=0.2, color='red', label='90% CI')
        axes[0, 0].set_title(f'{self.model_type} Price Paths')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Final price distribution
        axes[0, 1].hist(self.summary_stats['final_prices'], bins=50, 
                       alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(self.summary_stats['final_mean'], color='red', 
                          linestyle='--', linewidth=2, label='Mean')
        axes[0, 1].axvline(self.parameters['S0'], color='green', 
                          linestyle='--', linewidth=2, label='Current Price')
        axes[0, 1].set_title('Final Price Distribution')
        axes[0, 1].set_xlabel('Final Price')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Return distribution
        returns = (self.summary_stats['final_prices'] - self.parameters['S0']) / self.parameters['S0']
        axes[1, 0].hist(returns, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].axvline(np.mean(returns), color='red', 
                          linestyle='--', linewidth=2, label='Mean Return')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Price evolution statistics
        axes[1, 1].plot(time_axis, self.summary_stats['mean_path'], 
                       label='Mean', linewidth=2)
        axes[1, 1].plot(time_axis, self.summary_stats['median_path'], 
                       label='Median', linewidth=2)
        axes[1, 1].fill_between(time_axis,
                               self.summary_stats['mean_path'] - self.summary_stats['std_path'],
                               self.summary_stats['mean_path'] + self.summary_stats['std_path'],
                               alpha=0.3, label='±1 Std')
        axes[1, 1].set_title('Price Evolution Statistics')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Price')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print summary of simulation results."""
        if self.summary_stats is None:
            print("No simulation results available. Run simulation first.")
            return
        
        print(f"\n{self.model_type} Monte Carlo Simulation Summary")
        print("=" * 50)
        print(f"Number of simulations: {self.n_simulations:,}")
        print(f"Time horizon: {self.time_horizon} days")
        print(f"Current price: ${self.parameters['S0']:.2f}")
        
        print(f"\nFinal Price Statistics:")
        print(f"Mean: ${self.summary_stats['final_mean']:.2f}")
        print(f"Median: ${self.summary_stats['final_median']:.2f}")
        print(f"Standard deviation: ${self.summary_stats['final_std']:.2f}")
        print(f"5th percentile: ${self.summary_stats['lower_5'][-1]:.2f}")
        print(f"95th percentile: ${self.summary_stats['upper_95'][-1]:.2f}")
        
        print(f"\nProbabilities:")
        print(f"Probability of positive return: {self.summary_stats['prob_positive']:.2%}")
        
        for threshold, prob in self.summary_stats['prob_above_threshold'].items():
            print(f"Probability above ${threshold}: {prob:.2%}")

def main():
    """
    Main function to demonstrate Monte Carlo simulation with multiple stocks.
    """
    print("Monte Carlo Simulation Demo - Multiple Stocks")
    print("============================================")
    
    # List of stocks to analyze
    stocks = ["OPEN", "OPEN", "AMD", "SPY"]
    
    for stock_symbol in stocks:
        print(f"\n{'='*80}")
        print(f"ANALYZING {stock_symbol} STOCK")
        print(f"{'='*80}")
        
        # Load real stock data
        try:
            from data_loader import StockDataLoader
            
            data_loader = StockDataLoader()
            stock_data = data_loader.fetch_stock_data(stock_symbol, period="1y")  # Use 1 year for parameter estimation
            
            if stock_data is None or len(stock_data) < 100:
                raise ValueError(f"Insufficient {stock_symbol} stock data")

            print(f"Loaded {len(stock_data)} days of {stock_symbol} stock data")
            print(f"Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
            print(f"Price range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")
            
            # Use Close prices for Monte Carlo simulation
            price_series = stock_data['Close']
            current_price = price_series.iloc[-1]
            
            print(f"Current {stock_symbol} price: ${current_price:.2f}")
            
            # DEBUGGING: Check the actual data being passed to simulation
            print("--- Data being sent to the simulation ---")
            print(f"Price series length: {len(price_series)}")
            print(f"First 5 prices: {price_series.head().values}")
            print(f"Last 5 prices: {price_series.tail().values}")
            print(f"Price series type: {type(price_series)}")
            print("-----------------------------------------")
            
            # Test different Monte Carlo models
            models = ['GBM', 'Jump', 'Heston']
            
            for model_type in models:
                print(f"\n{'='*60}")
                print(f"Testing {model_type} Model on {stock_symbol} Stock")
                print(f"{'='*60}")
                
                # Initialize Monte Carlo model
                mc_model = MonteCarloForecaster(
                    model_type=model_type,
                    n_simulations=5000,
                    time_horizon=30
                )
                
                # Estimate parameters from stock data
                print(f"Estimating {model_type} parameters from {stock_symbol} data...")
                print(f"DEBUG: About to estimate parameters with price_series of length {len(price_series)}")
                print(f"DEBUG: Price series min/max: ${price_series.min():.2f} - ${price_series.max():.2f}")
                
                mc_model.estimate_parameters(price_series)
                
                print(f"DEBUG: After parameter estimation, S0 = ${mc_model.parameters.get('S0', 'NOT_SET'):.2f}")
                print(f"DEBUG: After parameter estimation, mu = {mc_model.parameters.get('mu', 'NOT_SET'):.4f}")
                print(f"DEBUG: After parameter estimation, sigma = {mc_model.parameters.get('sigma', 'NOT_SET'):.4f}")
                
                # Run simulation
                print("Running Monte Carlo simulations...")
                results = mc_model.run_simulation()
                
                # Print summary
                mc_model.print_summary()
                
                # Calculate VaR and ES
                var_es = mc_model.calculate_var_es()
                print(f"\n{stock_symbol} Risk Measures ({model_type}):")
                for measure, value in var_es.items():
                    print(f"{measure}: {value:.2%}")
                
                # Option pricing example
                strike_price = current_price * 1.05  # 5% out of the money
                option_result = mc_model.option_pricing(strike_price, 'call')
                print(f"\n{stock_symbol} Call Option ({model_type}, Strike=${strike_price:.2f}):")
                print(f"Option Price: ${option_result['option_price']:.2f}")
                print(f"95% CI: [${option_result['confidence_interval'][0]:.2f}, "
                      f"${option_result['confidence_interval'][1]:.2f}]")
                
                # Show forecast summary
                final_prices = results['price_paths'][:, -1]  # Last day prices from all simulations
                mean_forecast = np.mean(final_prices)
                std_forecast = np.std(final_prices)
                print(f"\n30-day {stock_symbol} Forecast ({model_type}):")
                print(f"Expected Price: ${mean_forecast:.2f}")
                print(f"Standard Deviation: ${std_forecast:.2f}")
                print(f"Price Change: {((mean_forecast - current_price) / current_price * 100):+.2f}%")
                print(f"95% Confidence Interval: [${np.percentile(final_prices, 2.5):.2f}, ${np.percentile(final_prices, 97.5):.2f}]")
        
        except Exception as e:
            print(f"Error loading {stock_symbol} data: {e}")
            print(f"Skipping {stock_symbol} and continuing with next stock...")
            continue
    
    print(f"\n{'='*80}")
    print("MULTI-STOCK ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
