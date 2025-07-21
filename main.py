import pandas as pd
import numpy as np
import logging
import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from models import ModelFactory, tune_all_models

# Option: Specify which models to use (by name, as in ModelFactory.get_available_models().keys())
MODEL_TO_USE = ['Random Forest', 'LightGBM', 'Ridge Regression', 'Lasso Regression', 
                'ElasticNet', 'Decision Tree', 'ExtraTreeRegressor', 'XGBoost', 
                'CatBoost', 'HistGradientBoosting', 'SVR', 'KNN'] 

# Setup logging
log_filename = f"logs/pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

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

    # 4. Model Creation (filtering out deep learning except ANN, and removing GAM, CERT/CatBoost)
    try:
        all_models = ModelFactory.get_available_models()
        # Remove deep learning models except ANN/MLPRegressor, and remove GAM, CERT, CatBoost
        exclude_models = {"RNN", "LSTM", "BI-LSTM", "GRU", "GAM", "CatBoost", "CERT"}
        eligible_models = {k: v for k, v in all_models.items() if k not in exclude_models}
        if MODEL_TO_USE:
            # Only use models specified in MODEL_TO_USE (if present in eligible_models)
            selected_models = {k: eligible_models[k] for k in MODEL_TO_USE if k in eligible_models}
        else:
            selected_models = eligible_models
        models = {name: cls(random_state=42) for name, cls in selected_models.items()}
        logging.info(f"Models to evaluate: {list(models.keys())}")
    except Exception as e:
        logging.error(f"Model creation failed: {e}")
        return

    # 5. Untuned Model Evaluation
    results = []
    actual_dict = {"Actual": y_test.values}
    for name, model in models.items():
        logging.info(f"Training and evaluating: {name}")
        try:
            pipeline = model.get_untuned_model(preprocessor)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            actual_dict[name] = y_pred
            results.append({
                "Model": name,
                "R²": round(r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4)
            })
            logging.info(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        except Exception as e:
            logging.error(f"{name} failed: {e}")
            continue
    pd.DataFrame(results).to_csv("Model_Results_UNTUNED.csv", index=False)

    # 6. Hyperparameter Tuning (optional, can be slow for all models)
    logging.info("Starting hyperparameter tuning (this may take a while)...")
    try:
        tuning_results = tune_all_models(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            models_to_tune=list(models.keys()),
            n_iter=10,  # Reduce for speed, increase for better search
            cv=3,
            n_jobs=-1,
            random_state=42
        )
    except Exception as e:
        logging.error(f"Hyperparameter tuning failed: {e}")
        tuning_results = {}

    # 7. Final Evaluation (Tuned Models)
    final_results = []
    actual_dict = {"Actual": y_test.values}
    for name, model in models.items():
        logging.info(f"Evaluating {name} (tuned/untuned)...")
        model_type = "Untuned"
        pipeline = None
        try:
            if name in tuning_results and "error" not in tuning_results[name]:
                best_params = tuning_results[name].get('best_params')
                if best_params is not None:
                    try:
                        pipeline = model.get_tuned_model(preprocessor, best_params=best_params)
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)
                        model_type = "Tuned"
                    except Exception as e:
                        logging.warning(f"Could not use tuned parameters for {name}: {e}. Using untuned model.")
            if pipeline is None:
                pipeline = model.get_untuned_model(preprocessor)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            actual_dict[f"{name} ({model_type})"] = y_pred
            final_results.append({
                "Model": f"{name}",
                "R²": round(r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4)
            })
            logging.info(f"{name} ({model_type}) - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        except Exception as e:
            logging.error(f"{name} ({model_type}) failed: {e}")
            continue
    pd.DataFrame(final_results).to_csv("Model_Results_TUNED.csv", index=False)
    logging.info(f"Pipeline run complete. See {log_filename} for details.")

if __name__ == "__main__":
    main() 

"""  
############ TODO ############
- check why models r2 getting worse after finetuning, fix that.
- try finetunning based on r2 score.
- add more metrics- mape, mbe, nmb
- validate the result by taking the parameter values and fitting and evaluating it seperately.
- also fix the imputer pipeline
- check if standardizations can work better, but do only after checking for above issues.


"""