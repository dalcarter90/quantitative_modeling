import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error (MAE)."""
    return mean_absolute_error(y_true, y_pred)

def evaluation_report(y_true, y_pred):
    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    return {
        'RMSE': rmse,
        'MAE': mae
    }
