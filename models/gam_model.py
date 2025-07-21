"""
GAM (Generalized Additive Model) implementation with hyperparameter tuning.
"""
from pygam import LinearGAM
from typing import Dict, Any
from .base_model import BaseModel

class GAMModel(BaseModel):
    """GAM (Generalized Additive Model) with hyperparameter tuning capabilities."""
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "GAM"
    def get_model(self):
        # LinearGAM does not accept random_state in its constructor
        return LinearGAM()
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__lam': [0.1, 1, 10, 100],
            'regressor__n_splines': [10, 20, 30],
            'regressor__max_iter': [100, 200, 500]
        }
    def get_default_params(self) -> Dict[str, Any]:
        # Remove random_state from default params as LinearGAM does not accept it
        return {
            'lam': 1.0,
            'n_splines': 20,
            'max_iter': 100
        }
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        return {
            'regressor__lam': [0.1, 1, 10],
            'regressor__n_splines': [10, 20],
        } 