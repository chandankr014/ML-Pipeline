"""
Model factory for creating and managing models in the pipeline.
"""

from typing import Dict, Any
from .random_forest_model import RandomForestModel
from .lightgbm_model import LightGBMModel
from .ridge_regression_model import RidgeRegressionModel
from .lasso_regression_model import LassoRegressionModel
from .elasticnet_model import ElasticNetModel
from .decision_tree_model import DecisionTreeModel
from .extratree_regressor_model import ExtraTreeRegressorModel
from .xgboost_model import XGBoostModel
from .catboost_model import CatBoostModel
from .hist_gradient_boosting_model import HistGradientBoostingModel
from .svr_model import SVRModel
from .knn_model import KNNModel

class ModelFactory:
    """Factory class for creating and managing models."""

    @staticmethod
    def get_available_models() -> Dict[str, Any]:
        """
        Get all available models.

        Returns:
            Dictionary mapping model names to model classes
        """
        return {
            "Random Forest": RandomForestModel,
            "LightGBM": LightGBMModel,
            "Ridge Regression": RidgeRegressionModel,
            "Lasso Regression": LassoRegressionModel,
            "ElasticNet": ElasticNetModel,
            "Decision Tree": DecisionTreeModel,
            "ExtraTreeRegressor": ExtraTreeRegressorModel,
            "XGBoost": XGBoostModel,
            "CatBoost": CatBoostModel,
            "HistGradientBoosting": HistGradientBoostingModel,
            "SVR": SVRModel,
            "KNN": KNNModel,
        }

    @staticmethod
    def create_model(model_name: str, random_state: int = 42):
        """
        Create a model instance by name.

        Args:
            model_name: Name of the model to create
            random_state: Random seed for reproducibility

        Returns:
            Model instance

        Raises:
            ValueError: If model name is not recognized
        """
        available_models = ModelFactory.get_available_models()

        if model_name not in available_models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(available_models.keys())}")

        model_class = available_models[model_name]
        return model_class(random_state=random_state)

    @staticmethod
    def create_all_models(random_state: int = 42) -> Dict[str, Any]:
        """
        Create instances of all available models.

        Args:
            random_state: Random seed for reproducibility

        Returns:
            Dictionary mapping model names to model instances
        """
        models = {}
        for model_name in ModelFactory.get_available_models().keys():
            models[model_name] = ModelFactory.create_model(model_name, random_state)

        return models