from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/lgbm_model_final.pkl")
THRESHOLD_PATH = Path("models/optimal_threshold.pkl")  # Fixed: Use pickle file in models/

# Define app FIRST before usage
app = FastAPI(title="Credit Scoring API", version="1.0.0")

class ClientFeatures(BaseModel):
    client_id: int = Field(..., description="Identifiant client")
    features: list[float] = Field(..., description="Vecteur de features déjà transformé")


def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        # Load directly from file since MLflow server is not available on Render
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Model loading failed: {exc}") from exc


def load_threshold(default: float = 0.5) -> float:
    try:
        if THRESHOLD_PATH.exists():
            return float(joblib.load(THRESHOLD_PATH))
    except Exception:
        pass  # Fallback to default if load fails
    return default


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model_stage": "Production"}


@app.post("/predict")
async def predict(payload: ClientFeatures) -> dict[str, Any]:
    model = load_model()
    threshold = load_threshold()
    
    # Reshape for prediction
    array = np.array(payload.features, dtype=float).reshape(1, -1)
    
    # Use predict_proba for probabilities (Class 1 is the positive class)
    # Check if model has predict_proba, otherwise fall back to predict
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(array)[0, 1])
    else:
        # Fallback for models without probability (e.g. some regressors or simple classifiers)
        proba = float(model.predict(array)[0])

    decision = int(proba >= threshold)
    
    return {
        "client_id": payload.client_id,
        "probability": proba,
        "decision": decision,
        "threshold": threshold,
    }


@app.post("/explain")
async def explain(payload: ClientFeatures) -> dict[str, Any]:
    """
    Returns SHAP values for the given features.
    For LightGBM, we use TreeExplainer for efficiency.
    Returns:
    - shap_values: List or array of SHAP values
    - base_value: The expected value (base line)
    - feature_names: List of feature names if available in model
    """
    model = load_model()
    features_array = np.array(payload.features).reshape(1, -1)
    
    # Use TreeExplainer for tree-based models (LightGBM, XGBoost, etc.)
    # It is much faster and accurate than KernelExplainer for these models.
    try:
        explainer = shap.TreeExplainer(model)
        shap_values_all = explainer.shap_values(features_array)
        
        # Handle binary classification case specifically
        # LightGBM binary: shap_values is often a list of [array_class0, array_class1]
        # We want class 1 (Positive/Default)
        if isinstance(shap_values_all, list) and len(shap_values_all) == 2:
            shap_values = shap_values_all[1][0]  # Take first row of class 1
            base_value = explainer.expected_value[1]
        else:
            # If it's not a list, it might be just the array (e.g. regression or different shap version)
            # Or if len != 2
            shap_values = shap_values_all[0] if len(shap_values_all.shape) > 1 else shap_values_all
            
            # expected_value might also be a list or scalar
            if isinstance(explainer.expected_value, list) or isinstance(explainer.expected_value, np.ndarray):
                 base_value = explainer.expected_value[-1] # Assume last is positive? Careful here.
                 if len(explainer.expected_value) == 2:
                     base_value = explainer.expected_value[1]
            else:
                base_value = explainer.expected_value

    except Exception as e:
        # Fallback to KernelExplainer if TreeExplainer fails
        print(f"TreeExplainer failed: {e}. Falling back to KernelExplainer.")
        background = np.zeros((1, len(payload.features))) # Minimal background
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(features_array)[0]
        base_value = explainer.expected_value

    # Extract feature names if possible
    feature_names = []
    if hasattr(model, "feature_name_"):
        feature_names = model.feature_name_
    elif hasattr(model, "feature_names_in_"):
        feature_names = model.feature_names_in_.tolist()

    return {
        "client_id": payload.client_id,
        "shap_values": shap_values.tolist(),
        "base_value": float(base_value),
        "feature_names": feature_names,
        "data": payload.features # Return data specifically for reconstruction
    }
