"""
MLPRegressor (ANN) model implementation with hyperparameter tuning.
"""
from sklearn.neural_network import MLPRegressor
from typing import Dict, Any
from .base_model import BaseModel

class MLPRegressorModel(BaseModel):
    """MLPRegressor (ANN) model with hyperparameter tuning capabilities."""
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "MLPRegressor"
    def get_model(self):
        return MLPRegressor(random_state=self.random_state, max_iter=1000)
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'regressor__activation': ['relu', 'tanh', 'logistic'],
            'regressor__solver': ['adam', 'lbfgs', 'sgd'],
            'regressor__alpha': [0.0001, 0.001, 0.01],
            'regressor__learning_rate': ['constant', 'adaptive'],
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'max_iter': 1000,
            'random_state': self.random_state
        }
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        return {
            'regressor__hidden_layer_sizes': [(50,), (100,)],
            'regressor__activation': ['relu', 'tanh'],
            'regressor__solver': ['adam'],
            'regressor__alpha': [0.0001, 0.001],
        } 