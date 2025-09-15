import numpy as np
import pandas as pd

def scenario_stress_test(
    S0,
    n_sims=5000,
    n_days=30,
    mu=None,
    sigma=None,
    custom_mu=None,
    custom_sigma=None,
    model='GBM',
    random_seed=42
):
    """
    Run scenario analysis by overriding drift (mu) and volatility (sigma).
    """
    np.random.seed(random_seed)
    dt = 1/252
    mu_used = custom_mu if custom_mu is not None else mu
    sigma_used = custom_sigma if custom_sigma is not None else sigma
    S = np.zeros((n_sims, n_days+1))
    S[:,0] = S0
    for t in range(1, n_days+1):
        Z = np.random.normal(size=n_sims)
        S[:,t] = S[:,t-1] * np.exp((mu_used - 0.5*sigma_used**2)*dt + sigma_used*np.sqrt(dt)*Z)
    return S
