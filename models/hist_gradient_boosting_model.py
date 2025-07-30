"""
HistGradientBoostingRegressor model implementation with hyperparameter tuning.
"""
from sklearn.ensemble import HistGradientBoostingRegressor
from typing import Dict, Any
from .base_model import BaseModel

class HistGradientBoostingModel(BaseModel):
    """HistGradientBoostingRegressor model with hyperparameter tuning capabilities."""
    
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "HistGradientBoosting"
    
    def get_model(self):
        return HistGradientBoostingRegressor(random_state=self.random_state)
    
    def get_param_distributions(self) -> Dict[str, Any]:
        """
        Return parameter distributions for RandomizedSearchCV (legacy - large grid).
        
        Returns:
            Dictionary of parameter distributions for HistGradientBoosting
        """
        return {
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'regressor__max_iter': [100, 200, 300],
            'regressor__max_depth': [3, 5, 7, None],
            'regressor__min_samples_leaf': [20, 30, 50],
            'regressor__l2_regularization': [0, 0.1, 1.0, 10.0],
            'regressor__max_bins': [255, 512]
        }
    
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid with reduced combinations.
        Target: 108 combinations (3×3×3×2×2 = 108)
        
        Returns:
            Dictionary of optimized parameters for HistGradientBoosting
        """
        return {
            'regressor__learning_rate': [0.01, 0.1, 0.2],        # 3 values
            'regressor__max_iter': [100, 200, 300],               # 3 values
            'regressor__max_depth': [3, 7, None],                 # 3 values
            'regressor__min_samples_leaf': [20, 50],              # 2 values
            'regressor__l2_regularization': [0, 1.0]              # 2 values
        }
        # Total: 3×3×3×2×2 = 108 combinations → Use GridSearchCV
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'learning_rate': 0.1,
            'max_iter': 100,
            'max_depth': None,
            'min_samples_leaf': 20,
            'l2_regularization': 0.0,
            'max_bins': 255,
            'random_state': self.random_state
        } 