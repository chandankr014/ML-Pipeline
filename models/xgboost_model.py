"""
XGBoost model implementation with hyperparameter tuning.
"""
from xgboost import XGBRegressor
from typing import Dict, Any
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost model with hyperparameter tuning capabilities."""
    
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "XGBoost"
    
    def get_model(self):
        return XGBRegressor(random_state=self.random_state, verbosity=0)
    
    def get_param_distributions(self) -> Dict[str, Any]:
        """
        Return parameter distributions for RandomizedSearchCV (legacy - large grid).
        
        Returns:
            Dictionary of parameter distributions for XGBoost
        """
        return {
            'regressor__n_estimators': [50, 100, 200, 300],
            'regressor__max_depth': [3, 5, 7, 10],
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'regressor__subsample': [0.6, 0.8, 1.0],
            'regressor__colsample_bytree': [0.6, 0.8, 1.0],
            'regressor__gamma': [0, 0.1, 0.2, 1],
            'regressor__reg_alpha': [0, 0.01, 0.1, 1],
            'regressor__reg_lambda': [0, 0.01, 0.1, 1]
        }
    
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid with reduced combinations.
        Target: 162 combinations (3×3×3×2×3 = 162)
        
        Returns:
            Dictionary of optimized parameters for XGBoost
        """
        return {
            'regressor__n_estimators': [100, 200, 300],          # 3 values
            'regressor__max_depth': [3, 5, 7],                   # 3 values
            'regressor__learning_rate': [0.01, 0.1, 0.2],        # 3 values
            'regressor__subsample': [0.8, 1.0],                  # 2 values
            'regressor__colsample_bytree': [0.8, 0.9, 1.0]       # 3 values
        }
        # Total: 3×3×3×2×3 = 162 combinations → Use RandomizedSearchCV with n_iter=50
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': self.random_state,
            'verbosity': 0
        } 