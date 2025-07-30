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
        Return parameter distributions for RandomizedSearchCV (legacy - large grid).
        
        Returns:
            Dictionary of parameter distributions for LightGBM
        """
        return {
            'regressor__n_estimators': [100, 200, 300, 500],
            'regressor__max_depth': [5, 7, 11, 15],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__num_leaves': [20, 31, 50, 100],
            'regressor__min_child_samples': [10, 25, 50],
            'regressor__subsample': [0.6, 0.8, 1.0],
            'regressor__colsample_bytree': [0.6, 0.8, 1.0],
            'regressor__objective': ['regression', 'regression_l1']
        }
    
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid with reduced combinations.
        Target: 216 combinations (3×3×3×2×2×2 = 216)
        
        Returns:
            Dictionary of optimized parameters for LightGBM
        """
        return {
            'regressor__n_estimators': [100, 300, 500],          # 3 values
            'regressor__max_depth': [5, 9, 13],                  # 3 values
            'regressor__learning_rate': [0.01, 0.1, 0.2],        # 3 values
            'regressor__num_leaves': [31, 100],                  # 2 values
            'regressor__subsample': [0.8, 1.0],                  # 2 values
            'regressor__colsample_bytree': [0.8, 1.0]            # 2 values
        }
        # Total: 3×3×3×2×2×2 = 216 combinations → Use GridSearchCV
    
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