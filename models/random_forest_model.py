"""
Random Forest model implementation with hyperparameter tuning.
"""

from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model with hyperparameter tuning capabilities."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Random Forest model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state)
        self.model_name = "Random Forest"
    
    def get_model(self):
        """Return Random Forest model instance."""
        return RandomForestRegressor(random_state=self.random_state)
    
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid for Random Forest (<100 combinations).
        """
        return {
            'regressor__n_estimators': [100, 300, 500],
            'regressor__max_depth': [None, 20, 40],
            'regressor__min_samples_split': [2, 10, 20],
            'regressor__min_samples_leaf': [1, 4, 8],
            'regressor__max_features': ['sqrt', 'log2'],
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

    # Keep get_param_distributions for backward compatibility, but recommend using get_optimized_param_grid
    
    def get_param_distributions(self) -> Dict[str, Any]:
        """
        Return parameter distributions for RandomizedSearchCV.
        
        Returns:
            Dictionary of parameter distributions for Random Forest
        """
        return {
            'regressor__n_estimators': [50, 100, 200, 300, 500],
            'regressor__max_depth': [None, 10, 20, 30, 40, 50],
            'regressor__min_samples_split': [2, 5, 10, 15, 20],
            'regressor__min_samples_leaf': [1, 2, 4, 6, 8],
            'regressor__max_features': ['sqrt', 'log2', None],
            'regressor__bootstrap': [True, False],
            'regressor__oob_score': [True, False],
            'regressor__criterion': ['squared_error', 'absolute_error', 'poisson']
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Return default parameters for Random Forest.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': False,
            'criterion': 'squared_error',
            'random_state': self.random_state
        } 