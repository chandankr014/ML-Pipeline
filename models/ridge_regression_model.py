"""
Ridge Regression model implementation with hyperparameter tuning.
"""
from sklearn.linear_model import Ridge
from typing import Dict, Any
from .base_model import BaseModel

class RidgeRegressionModel(BaseModel):
    """Ridge Regression model with hyperparameter tuning capabilities."""
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "Ridge Regression"
    def get_model(self):
        return Ridge(random_state=self.random_state)
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'regressor__fit_intercept': [True, False],
            'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'alpha': 1.0,
            'fit_intercept': True,
            'solver': 'auto',
            'random_state': self.random_state
        }

    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid for Ridge Regression (<100 combinations).
        """
        return {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
            'regressor__fit_intercept': [True, False],
            'regressor__solver': ['auto', 'sag'],
        }

    def get_search_strategy(self) -> str:
        param_count = self._calculate_param_combinations(self.get_optimized_param_grid())
        if param_count < 100:
            return 'grid'
        elif param_count < 1000:
            return 'random'
        else:
            return 'bayesian'

    def get_scoring_metric(self) -> str:
        return 'r2' 