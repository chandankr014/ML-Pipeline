"""
SVR model implementation with hyperparameter tuning.
"""
from sklearn.svm import SVR
from typing import Dict, Any
from .base_model import BaseModel

class SVRModel(BaseModel):
    """SVR model with hyperparameter tuning capabilities."""
    
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "SVR"
    
    def get_model(self):
        # SVR does not use random_state, but keep signature consistent
        return SVR()
    
    def get_param_distributions(self) -> Dict[str, Any]:
        """
        Return parameter distributions for RandomizedSearchCV (legacy - large grid).
        
        Returns:
            Dictionary of parameter distributions for SVR
        """
        return {
            'regressor__C': [0.1, 1, 10, 100],
            'regressor__epsilon': [0.01, 0.1, 0.2, 0.5],
            'regressor__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'regressor__degree': [2, 3, 4],
            'regressor__gamma': ['scale', 'auto']
        }
    
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid with reduced combinations.
        Target: 72 combinations (3×3×2×4×1 = 72)
        
        Returns:
            Dictionary of optimized parameters for SVR
        """
        return {
            'regressor__C': [1, 10, 100],                        # 3 values
            'regressor__epsilon': [0.01, 0.1, 0.5],              # 3 values
            'regressor__kernel': ['rbf', 'linear'],               # 2 values
            'regressor__gamma': ['scale', 'auto', 0.1, 1.0]      # 4 values
        }
        # Total: 3×3×2×4 = 72 combinations → Use GridSearchCV
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'C': 1.0,
            'epsilon': 0.1,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale'
        } 