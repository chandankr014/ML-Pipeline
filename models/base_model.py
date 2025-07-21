"""
Base model class for all ML models in the pipeline.
Provides common interface for hyperparameter tuning and model training.
"""

from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import time
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
        """Return parameter grid for GridSearchCV (legacy method)."""
        pass
    
    def get_optimized_param_grid(self) -> Dict[str, Any]:
        """
        Return optimized parameter grid with < 300 combinations.
        Override this method in subclasses for optimized grids.
        
        Returns:
            Dictionary of optimized parameters with reduced combinations
        """
        # Default implementation returns original grid
        # Subclasses should override this with optimized grids
        return self.get_param_distributions()
    
    def _calculate_param_combinations(self, param_grid: Dict[str, Any] = None) -> int:
        """
        Calculate total number of parameter combinations.
        
        Args:
            param_grid: Parameter grid to calculate combinations for
            
        Returns:
            Number of parameter combinations
        """
        if param_grid is None:
            param_grid = self.get_optimized_param_grid()
        
        total = 1
        for values in param_grid.values():
            total *= len(values)
        return total
    
    def get_search_strategy(self) -> str:
        """
        Determine search strategy based on parameter combinations.
        
        Returns:
            'grid' for < 300 combinations, 'random' otherwise
        """
        param_count = self._calculate_param_combinations()
        return 'grid' if param_count < 300 else 'random'
    
    def get_scoring_metric(self) -> str:
        """
        Return scoring metric for optimization.
        
        Returns:
            Scoring metric string
        """
        return 'r2'  # Use R² instead of RMSE for consistency
    
    def get_search_budget(self) -> Dict[str, Any]:
        """
        Return computational budget for hyperparameter search.
        
        Returns:
            Dictionary with max_time and n_iter budgets
        """
        param_count = self._calculate_param_combinations()
        
        if param_count < 300:
            return {'max_time': 1800, 'n_iter': None}  # 30 min for grid search
        elif param_count <= 1000:
            return {'max_time': 1800, 'n_iter': 100}   # 30 min, 100 iterations
        else:
            return {'max_time': 1800, 'n_iter': 50}    # 30 min, 50 iterations
    
    def tune_hyperparameters(self, 
                           X_train, 
                           y_train, 
                           preprocessor,
                           n_iter: int = None,
                           cv: int = 5,
                           n_jobs: int = -1,
                           random_state: int = 42) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using smart search strategy.
        
        Args:
            X_train: Training features
            y_train: Training targets
            preprocessor: Preprocessing pipeline
            n_iter: Number of parameter settings sampled (for RandomizedSearchCV)
            cv: Cross-validation folds
            n_jobs: Number of jobs to run in parallel
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing best parameters and score
        """
        start_time = time.time()
        
        # Create full pipeline with preprocessor
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', self.get_model())
        ])
        
        # Get optimized parameter grid and search strategy
        param_grid = self.get_optimized_param_grid()
        strategy = self.get_search_strategy()
        budget = self.get_search_budget()
        scoring = self.get_scoring_metric()
        
        # Override n_iter if provided
        if n_iter is not None:
            budget['n_iter'] = n_iter
        
        param_count = self._calculate_param_combinations(param_grid)
        print(f"Model: {getattr(self, 'model_name', self.__class__.__name__)}")
        print(f"Parameter combinations: {param_count:,}")
        print(f"Search strategy: {strategy}")
        print(f"Scoring metric: {scoring}")
        
        # Choose search method based on strategy
        if strategy == 'grid':
            print(f"Using GridSearchCV (combinations < 300)")
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
        else:
            n_iter_budget = budget['n_iter']
            print(f"Using RandomizedSearchCV with n_iter={n_iter_budget}")
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=n_iter_budget,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=1
            )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Store results
        self.best_params = search.best_params_
        self.best_score = search.best_score_  # R² score (higher is better)
        self.is_tuned = True
        
        elapsed_time = time.time() - start_time
        print(f"Best parameters: {self.best_params}")
        print(f"Best R² score: {self.best_score:.4f}")
        print(f"Tuning completed in {elapsed_time:.2f} seconds")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_estimator': search.best_estimator_,
            'search_strategy': strategy,
            'param_combinations': param_count,
            'elapsed_time': elapsed_time
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