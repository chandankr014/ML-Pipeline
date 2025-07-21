# Models Package

This package provides a modular structure for machine learning models with built-in hyperparameter tuning capabilities.

## Structure

```
models/
├── __init__.py              # Package initialization
├── base_model.py           # Abstract base class for all models
├── random_forest_model.py  # Random Forest implementation
├── lightgbm_model.py       # LightGBM implementation
├── model_factory.py        # Factory for creating models
├── tune_models.py          # Hyperparameter tuning utilities
└── README.md              # This file
```

## Features

- **Modular Design**: Each model is implemented as a separate class
- **Hyperparameter Tuning**: Built-in RandomizedSearchCV integration
- **Easy Extension**: Add new models by inheriting from BaseModel
- **Factory Pattern**: Easy model creation and management
- **Performance Comparison**: Compare tuned vs untuned models

## Usage

### Basic Usage

```python
from models import ModelFactory

# Create a specific model
rf_model = ModelFactory.create_model("Random Forest", random_state=42)

# Get untuned pipeline
pipeline = rf_model.get_untuned_model(preprocessor)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Hyperparameter Tuning

```python
from models import tune_all_models

# Tune all models
tuning_results = tune_all_models(
    X_train=X_train,
    y_train=y_train,
    preprocessor=preprocessor,
    n_iter=100,  # Number of iterations
    cv=5,        # Cross-validation folds
    random_state=42
)

# Use tuned model
rf_model = ModelFactory.create_model("Random Forest")
tuned_pipeline = rf_model.get_tuned_model(preprocessor)
tuned_pipeline.fit(X_train, y_train)
```

### Performance Comparison

```python
from models import compare_tuned_vs_untuned

# Compare tuned vs untuned performance
comparison_df = compare_tuned_vs_untuned(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    preprocessor=preprocessor,
    tuning_results=tuning_results
)
```

## Available Models

### Random Forest
- **File**: `random_forest_model.py`
- **Class**: `RandomForestModel`
- **Parameters**: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, oob_score, criterion

### LightGBM
- **File**: `lightgbm_model.py`
- **Class**: `LightGBMModel`
- **Parameters**: n_estimators, max_depth, learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda, boosting_type, objective

## Adding New Models

To add a new model:

1. Create a new file (e.g., `xgboost_model.py`)
2. Inherit from `BaseModel`
3. Implement required methods:
   - `get_model()`: Return the model instance
   - `get_param_distributions()`: Return parameter distributions for tuning
4. Add the model to `ModelFactory.get_available_models()`
5. Update `__init__.py` to include the new model

Example:

```python
from .base_model import BaseModel
from xgboost import XGBRegressor

class XGBoostModel(BaseModel):
    def get_model(self):
        return XGBRegressor(random_state=self.random_state)
    
    def get_param_distributions(self):
        return {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [3, 5, 7, 9],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            # ... more parameters
        }
```

## Configuration

### Tuning Parameters

- `n_iter`: Number of parameter settings sampled (default: 100)
- `cv`: Cross-validation folds (default: 5)
- `n_jobs`: Number of jobs to run in parallel (default: -1)
- `random_state`: Random seed for reproducibility (default: 42)

### Model Parameters

Each model has its own set of hyperparameters that can be tuned. See individual model files for the complete list of parameters.

## Dependencies

- scikit-learn
- lightgbm
- numpy
- pandas

## Examples

See `example_usage.py` and `Pipeline_with_Models.ipynb` for complete usage examples. 