"""
Inference module for loading the trained model and making predictions.
"""
from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Paths configuration
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model_final.pkl"
THRESHOLD_PATH = ROOT_DIR / "models" / "optimal_threshold.pkl"


def load_model():
    """Load the trained LightGBM model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)


def load_threshold(default: float = 0.5) -> float:
    """Load the optimal business threshold."""
    if THRESHOLD_PATH.exists():
        return float(joblib.load(THRESHOLD_PATH))
    return default


def predict_proba(features: np.ndarray) -> Dict[str, Any]:
    """Make a prediction with probability and business decision."""
    model = load_model()
    threshold = load_threshold()
    proba = model.predict_proba(features)[:, 1]
    decision = (proba >= threshold).astype(int)
    return {
        "probability": float(proba[0]),
        "decision": int(decision[0]),
        "threshold": threshold,
    }
