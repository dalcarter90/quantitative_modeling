"""
Pipeline Integration for Options Pricing Model
This module integrates the options pricing model with the existing Monte Carlo simulation pipeline.
"""

import pandas as pd
from montecarlo_pricing import calculate_option_prices

def load_simulation_data(file_path):
    """Load simulation data from a CSV file."""
    return pd.read_csv(file_path)

def integrate_pricing_model(simulation_data):
    """Integrate the options pricing model with the simulation data."""
    # Assuming simulation_data contains necessary columns for pricing
    option_prices = calculate_option_prices(simulation_data)
    return option_prices

if __name__ == "__main__":
    # Example usage
    simulation_data_path = '../data/sample_simulation_data.csv'
    simulation_data = load_simulation_data(simulation_data_path)
    option_prices = integrate_pricing_model(simulation_data)
    print(option_prices)  # Output the calculated option prices
