"""
Model training pipeline using LightGBM with business cost optimization.

This script:
1. Loads preprocessed features from artifacts/features/
2. Trains a LightGBM classifier with GridSearchCV
3. Optimizes the decision threshold for business cost (FN cost 10x FP)
4. Generates SHAP explanations for interpretability
5. Logs everything to MLflow and saves the model
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
from mlflow import sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import shap

from Src.models.custom_score import business_cost_score, optimal_threshold

# Paths configuration
ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts"
FEATURES_DIR = ARTIFACT_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"  # Output to models/ for API/Streamlit
PLOTS_DIR = MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_matrix(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and target for a given split."""
    X = pd.read_parquet(FEATURES_DIR / f"X_{split}.parquet").values
    y = pd.read_parquet(FEATURES_DIR / f"y_{split}.parquet")["target"].values
    return X, y


def load_sample_weights() -> np.ndarray:
    """Load sample weights for handling class imbalance."""
    weights_path = FEATURES_DIR / "sample_weights_train.parquet"
    if not weights_path.exists():
        raise FileNotFoundError("Sample weights missing. Run feature engineering first.")
    return pd.read_parquet(weights_path)["sample_weight"].values


def build_lgbm_estimator() -> GridSearchCV:
    """Build LightGBM classifier with GridSearchCV for hyperparameter tuning."""
    estimator = lgb.LGBMClassifier(
        objective="binary",
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
    )
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "num_leaves": [31, 63],
    }
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2,
    )
    return search


def log_shap_values(model, X_sample: np.ndarray) -> None:
    """Generate and log SHAP summary plot for global interpretability."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample[:200])
    
    # Handle binary classification output format
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (default)
    
    shap.summary_plot(shap_values, X_sample[:200], show=False)
    shap_path = PLOTS_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(shap_path, dpi=200)
    plt.close()
    mlflow.log_artifact(shap_path, artifact_path="explainability")
    print(f"SHAP summary saved to {shap_path}")


def train() -> None:
    """Main training pipeline with MLflow tracking."""
    # Configure MLflow
    tracking_uri = mlflow.get_tracking_uri() or "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit_scoring_prod")

    # Load data
    print("Loading training data...")
    X_train, y_train = load_matrix("train")
    X_valid, y_valid = load_matrix("valid")
    X_test, y_test = load_matrix("test")
    sample_weights = load_sample_weights()
    
    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # Build and train model
    search = build_lgbm_estimator()

    with mlflow.start_run(run_name="lightgbm_gridsearch"):
        print("Training LightGBM with GridSearchCV...")
        search.fit(X_train, y_train, sample_weight=sample_weights)
        best_model = search.best_estimator_
        
        # Log hyperparameters
        mlflow.log_params(search.best_params_)
        print(f"Best params: {search.best_params_}")

        # Evaluate on validation set
        valid_proba = best_model.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, valid_proba)
        
        # Optimize threshold for business cost
        threshold, cost_score = optimal_threshold(y_valid, valid_proba)
        
        mlflow.log_metric("valid_auc", valid_auc)
        mlflow.log_metric("optimal_threshold", threshold)
        mlflow.log_metric("business_cost_score", cost_score)
        print(f"Valid AUC: {valid_auc:.4f}, Optimal threshold: {threshold:.3f}")

        # Classification report
        y_valid_pred = (valid_proba >= threshold).astype(int)
        report = classification_report(y_valid, y_valid_pred, output_dict=True)
        mlflow.log_dict(report, "reports/valid_classification_report.json")

        # SHAP explanations
        print("Generating SHAP explanations...")
        log_shap_values(best_model, X_valid)

        # Save model to models/ directory (for API/Streamlit)
        model_path = MODELS_DIR / "lgbm_model_final.pkl"
        joblib.dump(best_model, model_path)
        
        # Also save threshold
        threshold_path = MODELS_DIR / "optimal_threshold.pkl"
        joblib.dump(threshold, threshold_path)
        
        # Log to MLflow registry
        mlflow_sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="credit_scoring_model",
        )
        mlflow.log_text(json.dumps({"threshold": float(threshold)}), "serving/threshold.json")

        # Final evaluation on holdout (valid + test)
        X_eval = np.vstack([X_valid, X_test])
        y_eval = np.concatenate([y_valid, y_test])
        test_proba = best_model.predict_proba(X_eval)[:, 1]
        test_auc = roc_auc_score(y_eval, test_proba)
        mlflow.log_metric("holdout_auc", test_auc)

        predictions = (test_proba >= threshold).astype(int)
        mlflow.log_metric("business_cost_holdout", business_cost_score(y_eval, predictions))

        print(f"\n✅ Model saved to {model_path}")
        print(f"✅ Threshold saved to {threshold_path}")
        print(f"✅ Model registered in MLflow as 'credit_scoring_model'")


if __name__ == "__main__":
    train()
