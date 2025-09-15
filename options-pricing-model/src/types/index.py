# src/types/index.py

"""
This file defines custom types and data structures used throughout the options pricing model project.
"""

from typing import NamedTuple, List

class OptionData(NamedTuple):
    strike_price: float
    expiration_date: str
    option_type: str  # 'call' or 'put'
    underlying_price: float
    volatility: float
    risk_free_rate: float

class PricingParameters(NamedTuple):
    num_simulations: int
    time_to_maturity: float  # in years
    steps_per_year: int
    discount_factor: float

def validate_option_data(option: OptionData) -> bool:
    """Validate the option data to ensure all fields are correctly set."""
    if option.strike_price <= 0:
        raise ValueError("Strike price must be positive.")
    if option.underlying_price <= 0:
        raise ValueError("Underlying price must be positive.")
    if option.volatility < 0:
        raise ValueError("Volatility cannot be negative.")
    if option.risk_free_rate < 0:
        raise ValueError("Risk-free rate cannot be negative.")
    return True

def validate_pricing_parameters(params: PricingParameters) -> bool:
    """Validate the pricing parameters."""
    if params.num_simulations <= 0:
        raise ValueError("Number of simulations must be positive.")
    if params.time_to_maturity <= 0:
        raise ValueError("Time to maturity must be positive.")
    if params.steps_per_year <= 0:
        raise ValueError("Steps per year must be positive.")
    if params.discount_factor <= 0:
        raise ValueError("Discount factor must be positive.")
    return True