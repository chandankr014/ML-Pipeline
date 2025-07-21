"""
ElasticNet regression model implementation with hyperparameter tuning.
"""
from sklearn.linear_model import ElasticNet
from typing import Dict, Any
from .base_model import BaseModel

class ElasticNetModel(BaseModel):
    """ElasticNet regression model with hyperparameter tuning capabilities."""
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "ElasticNet"
    def get_model(self):
        return ElasticNet(random_state=self.random_state, max_iter=10000)
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0],
            'regressor__fit_intercept': [True, False],
            'regressor__selection': ['cyclic', 'random']
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'fit_intercept': True,
            'selection': 'cyclic',
            'random_state': self.random_state,
            'max_iter': 10000
        } 