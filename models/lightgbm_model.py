"""
LightGBM model implementation with hyperparameter tuning.
"""

from lightgbm import LGBMRegressor
from typing import Dict, Any
from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM model with hyperparameter tuning capabilities."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize LightGBM model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state)
        self.model_name = "LightGBM"
    
    def get_model(self):
        """Return LightGBM model instance."""
        return LGBMRegressor(random_state=self.random_state, verbose=-1)
    
    def get_param_distributions(self) -> Dict[str, Any]:
        """
        Return parameter distributions for RandomizedSearchCV.
        
        Returns:
            Dictionary of parameter distributions for LightGBM
        """
        return {
            'regressor__n_estimators': [50, 100, 200, 300, 500],
            'regressor__max_depth': [3, 5, 7, 9, 11, 13, 15],
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
            'regressor__num_leaves': [10, 20, 31, 50, 100, 200],
            'regressor__min_child_samples': [10, 20, 30, 50, 100],
            'regressor__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'regressor__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'regressor__reg_alpha': [0, 0.01, 0.1, 1, 10],
            'regressor__reg_lambda': [0, 0.01, 0.1, 1, 10],
            'regressor__boosting_type': ['gbdt', 'dart'],
            'regressor__objective': ['regression', 'regression_l1', 'huber']
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Return default parameters for LightGBM.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'n_estimators': 100,
            'max_depth': 7,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'random_state': self.random_state,
            'verbose': -1
        } 

    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid for LightGBM (<100 combinations).
        """
        return {
            'regressor__n_estimators': [100, 300],
            'regressor__max_depth': [5, 10],
            'regressor__learning_rate': [0.01, 0.1, 1.0],
            'regressor__num_leaves': [31, 100],
            'regressor__min_child_samples': [10, 30],
            'regressor__subsample': [0.7, 1.0],
            'regressor__colsample_bytree': [0.7, 1.0],
            'regressor__reg_alpha': [0.01, 0.1, 1.0],
            'regressor__reg_lambda': [0.01, 0.1, 1.0],
            'regressor__boosting_type': ['gbdt'],
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