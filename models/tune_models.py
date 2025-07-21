"""
Utility script for hyperparameter tuning of all models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List
from .model_factory import ModelFactory


def tune_all_models(X_train, 
                   y_train, 
                   preprocessor,
                   models_to_tune: List[str] = None,
                   n_iter: int = 50,  # n_iter is ignored for GridSearchCV
                   cv: int = 5,
                   n_jobs: int = -1,
                   random_state: int = 42) -> Dict[str, Any]:
    """
    Tune hyperparameters for all models or specified models using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training targets
        preprocessor: Preprocessing pipeline
        models_to_tune: List of model names to tune. If None, tune all models
        n_iter: (ignored) Number of parameter settings sampled (for compatibility)
        cv: Cross-validation folds
        n_jobs: Number of jobs to run in parallel
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing tuning results for each model
    """
    # Get all available models
    available_models = ModelFactory.get_available_models()
    
    # If no specific models specified, tune all
    if models_to_tune is None:
        models_to_tune = list(available_models.keys())
    
    # Validate model names
    for model_name in models_to_tune:
        if model_name not in available_models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(available_models.keys())}")
    
    tuning_results = {}
    
    print(f"Starting hyperparameter tuning for {len(models_to_tune)} models...")
    print(f"Models to tune: {models_to_tune}")
    print(f"Number of iterations: {n_iter} (ignored for GridSearchCV)")
    print(f"Cross-validation folds: {cv}")
    print("-" * 50)
    
    for model_name in models_to_tune:
        print(f"\nTuning {model_name}...")
        
        # Create model instance
        model = ModelFactory.create_model(model_name, random_state)
        
        # Use model's optimized grid, strategy, and scoring
        param_grid = model.get_optimized_param_grid()
        strategy = model.get_search_strategy(param_grid)
        scoring = model.get_scoring_metric()
        budget = model.get_search_budget(param_grid)
        n_iter_search = budget['n_iter'] if strategy == 'random' else None
        try:
            results = model.tune_hyperparameters(
                X_train=X_train,
                y_train=y_train,
                preprocessor=preprocessor,
                n_iter=n_iter_search or 50,
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state
            )
            tuning_results[model_name] = {
                'model': model,
                'best_params': results['best_params'],
                'best_score': results['best_score'],
                'best_estimator': results['best_estimator'],
                'elapsed_time': results.get('elapsed_time', None)
            }
            print(f"✓ {model_name} tuning completed successfully!")
        except Exception as e:
            print(f"✗ Error tuning {model_name}: {str(e)}")
            tuning_results[model_name] = {
                'model': model,
                'error': str(e)
            }
    
    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 50)
    
    for model_name, results in tuning_results.items():
        if 'error' not in results:
            print(f"{model_name}:")
            print(f"  Best R2: {results['best_score']:.4f}")
            print(f"  Best params: {results['best_params']}")
            if results.get('elapsed_time'):
                print(f"  Tuning time: {results['elapsed_time']/60:.2f} min")
        else:
            print(f"{model_name}: ERROR - {results['error']}")
        print()
    
    return tuning_results


def compare_tuned_vs_untuned(X_train, 
                           y_train, 
                           X_test, 
                           y_test,
                           preprocessor,
                           tuning_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare performance of tuned vs untuned models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        preprocessor: Preprocessing pipeline
        tuning_results: Results from tune_all_models function
        
    Returns:
        DataFrame with comparison results
    """
    comparison_results = []
    
    for model_name, results in tuning_results.items():
        if 'error' in results:
            continue
            
        model = results['model']
        
        # Test untuned model
        untuned_pipeline = model.get_untuned_model(preprocessor)
        untuned_pipeline.fit(X_train, y_train)
        untuned_pred = untuned_pipeline.predict(X_test)
        
        untuned_r2 = r2_score(y_test, untuned_pred)
        untuned_rmse = np.sqrt(mean_squared_error(y_test, untuned_pred))
        untuned_mae = mean_absolute_error(y_test, untuned_pred)
        
        # Test tuned model
        tuned_pipeline = model.get_tuned_model(preprocessor)
        tuned_pipeline.fit(X_train, y_train)
        tuned_pred = tuned_pipeline.predict(X_test)
        
        tuned_r2 = r2_score(y_test, tuned_pred)
        tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
        tuned_mae = mean_absolute_error(y_test, tuned_pred)
        
        # Calculate percentage improvements
        r2_improvement_pct = ((tuned_r2 - untuned_r2) / abs(untuned_r2) * 100) if untuned_r2 != 0 else np.nan
        rmse_improvement_pct = ((untuned_rmse - tuned_rmse) / untuned_rmse * 100) if untuned_rmse != 0 else np.nan
        mae_improvement_pct = ((untuned_mae - tuned_mae) / untuned_mae * 100) if untuned_mae != 0 else np.nan

        comparison_results.append({
            'Model': model_name,
            'Untuned_R2': round(untuned_r2, 4),
            'Tuned_R2': round(tuned_r2, 4),
            'R2_Improvement(%)': round(r2_improvement_pct, 2),
            'Untuned_RMSE': round(untuned_rmse, 4),
            'Tuned_RMSE': round(tuned_rmse, 4),
            'RMSE_Improvement(%)': round(rmse_improvement_pct, 2),
            'Untuned_MAE': round(untuned_mae, 4),
            'Tuned_MAE': round(tuned_mae, 4),
            'MAE_Improvement(%)': round(mae_improvement_pct, 2)
        })
    
    return pd.DataFrame(comparison_results) 