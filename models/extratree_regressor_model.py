"""
ExtraTreeRegressor model implementation with hyperparameter tuning.
"""
from sklearn.tree import ExtraTreeRegressor
from typing import Dict, Any
from .base_model import BaseModel

class ExtraTreeRegressorModel(BaseModel):
    """ExtraTreeRegressor model with hyperparameter tuning capabilities."""
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "ExtraTreeRegressor"
    def get_model(self):
        return ExtraTreeRegressor(random_state=self.random_state)
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__max_depth': [None, 3, 5, 7, 10, 15, 20],
            'regressor__min_samples_split': [2, 5, 10, 20],
            'regressor__min_samples_leaf': [1, 2, 4, 8],
            'regressor__max_features': ['auto', None],
            'regressor__criterion': ['squared_error']
            # 'regressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        }
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid with reduced combinations for ExtraTreeRegressor.
        
        Returns:
            Dictionary of optimized parameters for ExtraTreeRegressor
        """
        return {
            'regressor__max_depth': [None, 5, 10, 15, 20],       # 5 values
            'regressor__min_samples_split': [2, 10, 20],         # 3 values
            'regressor__min_samples_leaf': [2, 5],               # 2 values
            'regressor__max_features': ['sqrt', None],           # 2 values
            'regressor__criterion': ['squared_error']            # 1 value
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'criterion': 'squared_error',
            'random_state': self.random_state
        } 