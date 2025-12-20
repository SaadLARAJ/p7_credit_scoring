from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
from mlflow import sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import shap

from Src.models.custom_score import business_cost_score, optimal_threshold

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
FEATURES_DIR = ARTIFACT_DIR / "features"
MODELS_DIR = ARTIFACT_DIR / "models"
PLOTS_DIR = ARTIFACT_DIR / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_matrix(split: str) -> tuple[np.ndarray, np.ndarray]:
    X = pd.read_parquet(FEATURES_DIR / f"X_{split}.parquet").values
    y = pd.read_parquet(FEATURES_DIR / f"y_{split}.parquet")["target"].values
    return X, y


def load_sample_weights() -> np.ndarray:
    weights_path = FEATURES_DIR / "sample_weights_train.parquet"
    if not weights_path.exists():
        raise FileNotFoundError("Sample weights missing. Run feature engineering first.")
    return pd.read_parquet(weights_path)["sample_weight"].values


def build_estimator() -> GridSearchCV:
    estimator = GradientBoostingClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4],
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


def log_shap_values(model, X_sample) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample[:200])
    shap.summary_plot(shap_values, X_sample[:200], show=False)
    shap_path = PLOTS_DIR / "shap_summary.png"
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(shap_path, dpi=200)
    mlflow.log_artifact(shap_path, artifact_path="explainability")


def train() -> None:
    tracking_uri = mlflow.get_tracking_uri() or "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit_scoring_prod")

    X_train, y_train = load_matrix("train")
    X_valid, y_valid = load_matrix("valid")
    X_test, y_test = load_matrix("test")
    sample_weights = load_sample_weights()

    search = build_estimator()

    with mlflow.start_run(run_name="gradient_boosting_gridsearch"):
        search.fit(X_train, y_train, sample_weight=sample_weights)
        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)

        valid_proba = best_model.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, valid_proba)
        threshold, cost_score = optimal_threshold(y_valid, valid_proba)
        mlflow.log_metric("valid_auc", valid_auc)
        mlflow.log_metric("optimal_threshold", threshold)
        mlflow.log_metric("business_cost_score", cost_score)

        y_valid_pred = (valid_proba >= threshold).astype(int)
        report = classification_report(y_valid, y_valid_pred, output_dict=True)
        mlflow.log_dict(report, "reports/valid_classification_report.json")

        log_shap_values(best_model, X_valid)

        model_path = MODELS_DIR / "gradient_boosting.joblib"
        joblib.dump(best_model, model_path)
        mlflow_sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="credit_scoring_model",
        )

        threshold_data = {"threshold": float(threshold)}
        threshold_path = MODELS_DIR / "threshold.json"
        threshold_path.write_text(json.dumps(threshold_data))
        mlflow.log_text(json.dumps(threshold_data), "serving/threshold.json")

        X_eval = np.vstack([X_valid, X_test])
        y_eval = np.concatenate([y_valid, y_test])
        test_proba = best_model.predict_proba(X_eval)[:, 1]
        test_auc = roc_auc_score(y_eval, test_proba)
        mlflow.log_metric("holdout_auc", test_auc)

        predictions = (test_proba >= threshold).astype(int)
        mlflow.log_metric("business_cost_holdout", business_cost_score(y_eval, predictions))

        print(f"Model saved to {model_path} and registered in MLflow.")


if __name__ == "__main__":
    train()
