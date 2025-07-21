"""
Base model class for all ML models in the pipeline.
Provides common interface for hyperparameter tuning and model training.
"""

from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Dict, Any, Optional


class BaseModel(ABC):
    """Abstract base class for all models in the pipeline."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the base model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.best_score = None
        self.is_tuned = False
        
    @abstractmethod
    def get_model(self):
        """Return the base model instance."""
        pass
    
    @abstractmethod
    def get_param_distributions(self) -> Dict[str, Any]:
        """Return parameter grid for GridSearchCV."""
        pass
    
    def tune_hyperparameters(self, 
                           X_train, 
                           y_train, 
                           preprocessor,
                           n_iter: int = 100,  # n_iter is not used in GridSearchCV, but keep for compatibility
                           cv: int = 5,
                           n_jobs: int = -1,
                           random_state: int = 42) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training targets
            preprocessor: Preprocessing pipeline
            n_iter: (ignored) Number of parameter settings sampled (for compatibility)
            cv: Cross-validation folds
            n_jobs: Number of jobs to run in parallel
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing best parameters and score
        """
        # Create full pipeline with preprocessor
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', self.get_model())
        ])
        
        # Get parameter grid
        param_grid = self.get_param_distributions()
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.best_params = grid_search.best_params_
        self.best_score = np.sqrt(-grid_search.best_score_)
        self.is_tuned = True
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best RMSE score: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_estimator': grid_search.best_estimator_
        }
    
    def get_tuned_model(self, preprocessor, best_params=None):
        """
        Get the tuned model with best parameters.
        
        Args:
            preprocessor: Preprocessing pipeline
            best_params: (Optional) Dict of best parameters to use (from tuning_results)
            
        Returns:
            Pipeline with tuned model
        """
        # Use provided best_params if given, otherwise use self.best_params
        params = best_params if best_params is not None else self.best_params
        if params is None:
            raise ValueError("Model must be tuned first or best_params must be provided. Call tune_hyperparameters() or pass best_params.")
        
        # Create model with best parameters
        tuned_model = self.get_model()
        estimator_params = {k.replace('regressor__', ''): v for k, v in params.items() if k.startswith('regressor__')}
        tuned_model.set_params(**estimator_params)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', tuned_model)
        ])
        
        return pipeline
    
    def get_untuned_model(self, preprocessor):
        """
        Get the untuned model with default parameters.
        
        Args:
            preprocessor: Preprocessing pipeline
            
        Returns:
            Pipeline with untuned model
        """
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', self.get_model())
        ])
        
        return pipeline 