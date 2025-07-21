import pandas as pd
import numpy as np
import logging
import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from models import ModelFactory, tune_all_models, compare_tuned_vs_untuned

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def main():
    try:
        # 1. Data Loading
        FILE = "data/mumbai_data_processed.csv"
        USE_COLS = ['year','month','WS','WD','SP','BLH','TCC','T2M','D2M','RH','SSR','TP', 'PM25_WUstl']
        df = pd.read_csv(FILE)
        if 'PM25_clean' in df.columns:
            df.loc[df['year'] == 2024, 'PM25_WUstl'] = df.loc[df['year'] == 2024, 'PM25_clean']
        df = df[USE_COLS]
        TARGET = 'PM25_WUstl'

        # Using a smaller subset for faster testing
        print(f"Original dataframe shape: {df.shape}")
        sample_size = min(100, len(df))
        df = df.sample(n=sample_size, random_state=42)

        X = df.drop(columns=TARGET)
        y = df[TARGET]
        logging.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        return

    # 2. Preprocessing
    try:
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        logging.info(f"Preprocessing pipeline created. Numeric: {numeric_features}, Categorical: {categorical_features}")
    except Exception as e:
        logging.error(f"Preprocessing setup failed: {e}")
        return

    # 3. Data Split
    try:
        train_mask = df["year"] < 2024
        test_mask = df["year"] == 2024
        X_train = df.loc[train_mask].drop(columns=[TARGET])
        y_train = df.loc[train_mask, TARGET]
        X_test = df.loc[test_mask].drop(columns=[TARGET])
        y_test = df.loc[test_mask, TARGET]
        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logging.info(f"Train shape: {y_train.shape}, Test shape: {y_test.shape}")
    except Exception as e:
        logging.error(f"Data split failed: {e}")
        return

    # 4. Model Tuning
    models_to_test = ['Random Forest']
    logging.info(f"Starting hyperparameter tuning for: {models_to_test}")

    # Create temporary test models with reduced search budget
    test_models = {}
    for model_name in models_to_test:
        original_model_class = ModelFactory.get_available_models()[model_name]

        class TestModel(original_model_class):
            def get_search_budget(self) -> Dict[str, int]:
                return {'n_iter': 1} # Reduced for testing

        ModelFactory.register_model(f"Test_{model_name}", TestModel)
        test_models[f"Test_{model_name}"] = TestModel

    tuning_results = tune_all_models(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        models_to_tune=list(test_models.keys()),
        cv=2, # Reduced CV for testing
        n_jobs=-1,
        random_state=42
    )

    # Map results back to original model names
    original_tuning_results = {}
    for test_model_name, results in tuning_results.items():
        original_name = test_model_name.replace("Test_", "")
        original_tuning_results[original_name] = results
    tuning_results = original_tuning_results

    # 5. Compare and Validate
    if tuning_results:
        logging.info("Comparing tuned vs. untuned models...")
        comparison_df = compare_tuned_vs_untuned(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            preprocessor=preprocessor,
            tuning_results=tuning_results
        )
        print("\n" + "="*50)
        print("TUNING COMPARISON RESULTS")
        print("="*50)
        print(comparison_df)

        # Basic validation checks
        for model in models_to_test:
            if model in comparison_df['Model'].values:
                r2_tuned = comparison_df[comparison_df['Model'] == model]['Tuned_R2'].values[0]
                assert r2_tuned > -np.inf, f"{model} R2 score is invalid."
            else:
                logging.warning(f"{model} not in comparison results.")

        logging.info("Test script finished successfully.")
    else:
        logging.error("Tuning failed, no results to compare.")

if __name__ == "__main__":
    main()
