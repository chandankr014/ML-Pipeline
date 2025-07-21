"""
Base model class for all ML models in the pipeline.
Provides common interface for hyperparameter tuning and model training.
"""

from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import time
from typing import Dict, Any, Optional, Union, List


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
        """Return parameter grid for hyperparameter tuning."""
        pass

    @abstractmethod
    def get_search_strategy(self) -> str:
        """Return search strategy ('grid', 'random', 'bayesian')."""
        pass
    
    @abstractmethod
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """Return optimized parameter grid."""
        pass

    @abstractmethod
    def get_search_budget(self) -> Dict[str, int]:
        """Return search budget (e.g., n_iter, timeout)."""
        pass

    @abstractmethod
    def get_scoring_metric(self) -> Union[str, List[str]]:
        """Return scoring metric(s) for optimization."""
        pass

    def _calculate_param_combinations(self, param_grid: Dict[str, Any]) -> int:
        """Calculate the total number of parameter combinations."""
        if not param_grid:
            return 0

        # Calculate the number of combinations
        combinations = 1
        for key, values in param_grid.items():
            combinations *= len(values)

        return combinations

    def tune_hyperparameters(self,
                               X_train,
                               y_train,
                               preprocessor,
                               cv: int = 5,
                               n_jobs: int = -1,
                               random_state: int = 42) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using the specified search strategy.
        
        Args:
            X_train: Training features
            y_train: Training targets
            preprocessor: Preprocessing pipeline
            cv: Cross-validation folds
            n_jobs: Number of jobs to run in parallel
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing best parameters, score, and estimator
        """
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', self.get_model())
        ])

        search_strategy = self.get_search_strategy()
        param_grid = self.get_optimized_param_grid()
        search_budget = self.get_search_budget()
        scoring = self.get_scoring_metric()

        searcher = None
        start_time = time.time()
        
        print(f"Using {search_strategy.capitalize()}SearchCV...")
        
        if search_strategy == 'grid':
            searcher = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
        elif search_strategy == 'random':
            searcher = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=search_budget.get('n_iter', 10),
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        elif search_strategy == 'bayesian':
            # Placeholder for Optuna integration
            print("Bayesian search (Optuna) is not yet implemented. Falling back to RandomizedSearch.")
            searcher = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=search_budget.get('n_iter', 20),
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        else:
            raise ValueError(f"Unsupported search strategy: {search_strategy}")

        # Fit the searcher
        searcher.fit(X_train, y_train)
        
        end_time = time.time()
        tuning_duration = end_time - start_time
        
        # Store results
        self.best_params = searcher.best_params_
        self.best_score = searcher.best_score_
        self.is_tuned = True
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best score ({scoring}): {self.best_score:.4f}")
        print(f"Tuning duration: {tuning_duration:.2f} seconds")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_estimator': searcher.best_estimator_,
            'tuning_duration': tuning_duration
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