"""
Utility script for hyperparameter tuning of all models with smart search strategy.
"""

import pandas as pd
import numpy as np
import os
import json
import time
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List
from .model_factory import ModelFactory


def tune_all_models(
                   CITY,
                   X_train, 
                   y_train, 
                   preprocessor,
                   models_to_tune: List[str] = None,
                   n_iter: int = None,
                   cv: int = 5,
                   n_jobs: int = -1,
                   random_state: int = 42) -> Dict[str, Any]:
    """
    Tune hyperparameters for all models using smart search strategy.
    
    Args:
        X_train: Training features
        y_train: Training targets
        preprocessor: Preprocessing pipeline
        models_to_tune: List of model names to tune. If None, tune all models
        n_iter: Number of parameter settings sampled (for RandomizedSearchCV)
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
    total_start_time = time.time()
    
    print("=" * 60)
    print("SMART HYPERPARAMETER TUNING STARTED")
    print("=" * 60)
    print(f"Models to tune: {models_to_tune}")
    print(f"Cross-validation folds: {cv}")
    print(f"Scoring metric: R² (higher is better)")
    print("-" * 60)
    
    for i, model_name in enumerate(models_to_tune, 1):
        print(f"\n[{i}/{len(models_to_tune)}] Tuning {model_name}...")
        
        # Create model instance
        model = ModelFactory.create_model(model_name, random_state)
        
        # This section uses methods defined in the base model class (see models/base_model.py):
        # - _calculate_param_combinations: calculates the number of parameter combinations
        # - get_search_strategy: returns 'grid' or 'random' based on param count
        # - get_search_budget: returns dict with 'max_time' and 'n_iter'
        # param_count = model._calculate_param_combinations()
        # score_metric = model.get_scoring_metric()
        

        # print(f"  Parameter combinations: {param_count:,}")
        # print(f"  Search strategy: {strategy.upper()}")
        # print(f"  Scoring metric: {score_metric.upper()}")
        
        """ FOR FULL PARAM GRID """
        # opt_param_grid = model.get_param_distributions()
        # print(f"  Full parameter distribution: {opt_param_grid}") 

        """ FOR OPTIMIZED PARAM GRID """
        opt_param_grid = model.get_optimized_param_grid()
        print(f"  Optimized parameter grid: {opt_param_grid}") 
        
        # PERFORM FINETUNING 
        try:
            results = model.tune_hyperparameters(
                X_train=X_train,
                y_train=y_train,
                preprocessor=preprocessor,
                n_iter=n_iter,
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state
            )
            
            tuning_results[model_name] = {
                'model': model,
                'best_params': results['best_params'],
                'best_score': results['best_score'],
                'best_estimator': results['best_estimator'],
                'search_strategy': results['search_strategy'],
                'param_combinations': results['param_combinations'],
                'elapsed_time': results['elapsed_time']
            }
            
            print(f"  ✓ {model_name} tuning completed successfully!")
            print(f"  ✓ Best R² score: {results['best_score']:.4f}")
            print(f"  ✓ Time elapsed: {results['elapsed_time']:.2f} seconds")
            
        except Exception as e:
            print(f"  ✗ Error tuning {model_name}: {str(e)}")
            tuning_results[model_name] = {
                'model': model,
                'error': str(e)
            }
    
    total_elapsed_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 60)
    
    successful_models = 0
    total_combinations = 0
    total_model_time = 0
    
    for model_name, results in tuning_results.items():
        if 'error' not in results:
            successful_models += 1
            total_combinations += results['param_combinations']
            total_model_time += results['elapsed_time']
            
            print(f"{model_name}:")
            print(f"  Strategy: {results['search_strategy'].upper()}")
            print(f"  Combinations: {results['param_combinations']:,}")
            print(f"  Best R² score: {results['best_score']:.4f}")
            print(f"  Time: {results['elapsed_time']:.2f}s")
            print(f"  Best params: {results['best_params']}")
        else:
            print(f"{model_name}: ERROR - {results['error']}")
        print()
    
    print(f"Successfully tuned models: {successful_models}/{len(models_to_tune)}")
    print(f"Total parameter combinations: {total_combinations:,}")
    print(f"Total tuning time: {total_elapsed_time:.2f} seconds")
    print(f"Average time per model: {total_model_time/max(successful_models, 1):.2f} seconds")
    print(f"Smart approach tested: {total_combinations:,} combinations")
    
    json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'json')
    os.makedirs(json_dir, exist_ok=True)
    # Prepare results for JSON serialization (remove non-serializable objects)
    def serialize_results(results):
        serializable = {}
        for model_name, res in results.items():
            serializable[model_name] = {}
            for k, v in res.items():
                if k in ['model', 'best_estimator']:
                    # Store string representation only
                    serializable[model_name][k] = str(v)
                else:
                    try:
                        json.dumps(v)
                        serializable[model_name][k] = v
                    except TypeError:
                        serializable[model_name][k] = str(v)
        return serializable

    serializable_results = serialize_results(tuning_results)
    json_path = os.path.join(json_dir, f"tuning_results_{CITY}.json")
    with open(json_path, "w") as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Tuning results saved to {json_path}")
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
    
    print("\n" + "=" * 60)
    print("TUNED vs UNTUNED MODEL COMPARISON")
    print("=" * 60)
    
    for model_name, results in tuning_results.items():
        if 'error' in results:
            continue
            
        print(f"\nEvaluating {model_name}...")
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
            'Search_Strategy': results['search_strategy'].upper(),
            'Param_Combinations': results['param_combinations'],
            'Tuning_Time(s)': round(results['elapsed_time'], 2),
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
        
        print(f"  Untuned R²: {untuned_r2:.4f} | Tuned R²: {tuned_r2:.4f} | Improvement: {r2_improvement_pct:.2f}%")
    
    df = pd.DataFrame(comparison_results)
    df.to_csv("results/comparison_results.csv", index=False)
    print(f"\nComparison completed for {len(comparison_results)} models.")
    return df


def generate_performance_report(tuning_results: Dict[str, Any], 
                              comparison_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Generate a comprehensive performance report.
    
    Args:
        tuning_results: Results from tune_all_models function
        comparison_df: DataFrame from compare_tuned_vs_untuned function
        
    Returns:
        Dictionary with performance metrics and summary
    """
    report = {
        'summary': {},
        'model_details': {},
        'computational_savings': {}
    }
    
    # Calculate summary statistics
    successful_models = [r for r in tuning_results.values() if 'error' not in r]
    total_combinations = sum(r['param_combinations'] for r in successful_models)
    total_time = sum(r['elapsed_time'] for r in successful_models)
    avg_time = total_time / len(successful_models) if successful_models else 0
    
    report['summary'] = {
        'total_models_tuned': len(successful_models),
        'total_parameter_combinations': total_combinations,
        'total_tuning_time_seconds': round(total_time, 2),
        'average_time_per_model_seconds': round(avg_time, 2),
        'max_time_per_model_seconds': max([r['elapsed_time'] for r in successful_models]) if successful_models else 0
    }
    
    # Model-specific details
    for model_name, results in tuning_results.items():
        if 'error' not in results:
            report['model_details'][model_name] = {
                'search_strategy': results['search_strategy'],
                'parameter_combinations': results['param_combinations'],
                'best_r2_score': round(results['best_score'], 4),
                'tuning_time_seconds': round(results['elapsed_time'], 2),
                'best_parameters': results['best_params']
            }
    
    # Computational savings
    original_combinations = 23691744  # From previous calculation
    reduction_pct = ((original_combinations - total_combinations) / original_combinations) * 100
    
    report['computational_savings'] = {
        'original_combinations': original_combinations,
        'optimized_combinations': total_combinations,
        'reduction_percentage': round(reduction_pct, 1),
        'speedup_factor': round(original_combinations / total_combinations, 1) if total_combinations > 0 else 0
    }
    with open("results/performance_report.json", "w") as f:
        json.dump(report, f, indent=4)
    return report 