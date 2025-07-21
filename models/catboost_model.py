"""
CatBoostRegressor model implementation with hyperparameter tuning.
"""
from catboost import CatBoostRegressor
from typing import Dict, Any
from .base_model import BaseModel

class CatBoostModel(BaseModel):
    """CatBoostRegressor model with hyperparameter tuning capabilities."""
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "CatBoost"
    def get_model(self):
        return CatBoostRegressor(random_state=self.random_state, verbose=0)
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__iterations': [100, 200, 300],
            'regressor__depth': [3, 5, 7, 10],
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'regressor__l2_leaf_reg': [1, 3, 5, 7, 9],
            'regressor__bagging_temperature': [0, 1, 5, 10],
            'regressor__border_count': [32, 64, 128]
        }
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        return {
            'regressor__iterations': [100, 300],
            'regressor__depth': [3, 7],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__l2_leaf_reg': [1, 5],
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            'bagging_temperature': 1,
            'border_count': 64,
            'random_state': self.random_state,
            'verbose': 0
        } 