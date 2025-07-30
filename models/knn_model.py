"""
KNN Regressor model implementation with hyperparameter tuning.
"""
from sklearn.neighbors import KNeighborsRegressor
from typing import Dict, Any
from .base_model import BaseModel

class KNNModel(BaseModel):
    """KNN Regressor model with hyperparameter tuning capabilities."""
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "KNN"
    def get_model(self):
        return KNeighborsRegressor()
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__n_neighbors': [3, 5, 7, 9, 11],
            'regressor__weights': ['uniform', 'distance'],
            'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'regressor__leaf_size': [10, 20, 30, 40, 50],
            'regressor__p': [1, 2]
        }
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        return {
            'regressor__n_neighbors': [3, 7, 11, 15],
            'regressor__weights': ['uniform', 'distance'],
            'regressor__algorithm': ['auto', 'ball_tree'],
            'regressor__leaf_size': [10, 20, 30, 50],
            'regressor__p': [1, 2]
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2
        } 