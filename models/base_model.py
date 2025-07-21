"""
Base model class for all ML models in the pipeline.
Provides common interface for hyperparameter tuning and model training.
"""

from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from typing import Dict, Any, Optional
import time


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
    
    @abstractmethod
    def get_search_strategy(self) -> str:
        """
        Return the search strategy: 'grid', 'random', or 'bayesian'.
        """
        pass

    @abstractmethod
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return the optimized (reduced) parameter grid for efficient search.
        """
        pass

    def get_search_budget(self) -> Dict[str, Any]:
        """
        Return search budget as a dict, e.g., {'max_time': 1800, 'n_iter': 50}.
        """
        return {'max_time': 1800, 'n_iter': 50}

    def get_scoring_metric(self) -> str:
        """
        Return the scoring metric for optimization (default: 'r2').
        """
        return 'r2'

    def _calculate_param_combinations(self, param_grid: Dict[str, Any]) -> int:
        """
        Calculate the number of parameter combinations in a grid.
        """
        from functools import reduce
        from operator import mul
        sizes = [len(v) for v in param_grid.values() if isinstance(v, list)]
        return reduce(mul, sizes, 1) if sizes else 0

    def tune_hyperparameters(self, 
                           X_train, 
                           y_train, 
                           preprocessor,
                           n_iter: int = 100,
                           cv: int = 5,
                           n_jobs: int = -1,
                           random_state: int = 42) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using the most efficient strategy.
        """
        try:
            import optuna
            from optuna.integration import OptunaSearchCV
            optuna_available = True
        except ImportError:
            optuna_available = False
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', self.get_model())
        ])
        param_grid = self.get_optimized_param_grid()
        search_strategy = self.get_search_strategy()
        scoring = self.get_scoring_metric()
        budget = self.get_search_budget()
        max_time = budget.get('max_time', 1800)
        n_iter_budget = budget.get('n_iter', n_iter)
        param_count = self._calculate_param_combinations(param_grid)
        search = None
        if search_strategy == 'grid':
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
        elif search_strategy == 'random':
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=min(n_iter_budget, param_count),
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=1
            )
        elif search_strategy == 'bayesian' and optuna_available:
            search = OptunaSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_trials=n_iter_budget,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=random_state,
                timeout=max_time,
                verbose=1
            )
        else:
            # fallback to random search
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=min(n_iter_budget, param_count),
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=1
            )
        start_time = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.is_tuned = True
        print(f"Best parameters: {self.best_params}")
        print(f"Best {scoring} score: {self.best_score:.4f}")
        print(f"Tuning time: {elapsed:.1f} seconds")
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_estimator': search.best_estimator_,
            'tuning_time': elapsed
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