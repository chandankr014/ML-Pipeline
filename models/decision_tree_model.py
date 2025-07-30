"""
Decision Tree Regressor model implementation with hyperparameter tuning.
"""
from sklearn.tree import DecisionTreeRegressor
from typing import Dict, Any
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    """Decision Tree Regressor model with hyperparameter tuning capabilities."""
    
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "Decision Tree"
    
    def get_model(self):
        return DecisionTreeRegressor(random_state=self.random_state)
    
    def get_param_distributions(self) -> Dict[str, Any]:
        """
        Return parameter distributions for RandomizedSearchCV (legacy - large grid).
        
        Returns:
            Dictionary of parameter distributions for Decision Tree
        """
        return {
            'regressor__max_depth': [None, 5, 10, 15, 20, 30],
            'regressor__min_samples_split': [2, 5, 10, 20],
            'regressor__min_samples_leaf': [1, 2, 4, 8],
            'regressor__max_features': ['auto', None],
            # 'regressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
            'regressor__criterion': ['squared_error', 'absolute_error']
        }
    
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid with reduced combinations.
        Target: 120 combinations (5×3×2×2×2 = 120)
        
        Returns:
            Dictionary of optimized parameters for Decision Tree
        """
        return {
            'regressor__max_depth': [None, 5, 10, 15, 20],       # 5 values
            'regressor__min_samples_split': [2, 10, 20],         # 3 values
            'regressor__min_samples_leaf': [1, 4],               # 2 values
            'regressor__max_features': ['sqrt', None],           # 2 values
            'regressor__criterion': ['squared_error', 'absolute_error']  # 2 values
        }
        # Total: 5×3×2×2×2 = 120 combinations → Use GridSearchCV
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'criterion': 'squared_error',
            'random_state': self.random_state
        } 