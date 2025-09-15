import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class RandomForestForecaster:
    """
    Random Forest regressor for time series forecasting.
    """
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.fitted = False

    def prepare_data(self, features, target_col):
        X = features.drop(columns=[target_col]).values
        y = features[target_col].values
        return X, y

    def fit(self, features, target_col):
        X, y = self.prepare_data(features, target_col)
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, features):
        X = features.values if isinstance(features, pd.DataFrame) else features
        return self.model.predict(X)

    def feature_importances(self):
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        return self.model.feature_importances_
