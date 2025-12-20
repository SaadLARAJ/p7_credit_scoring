from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np

THRESHOLD_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "threshold.json"


def load_model(stage: str = "Production"):
    model_uri = f"models:/credit_scoring_model/{stage}"
    return mlflow.pyfunc.load_model(model_uri)


def load_threshold(default: float = 0.5) -> float:
    if THRESHOLD_PATH.exists():
        return json.loads(THRESHOLD_PATH.read_text()).get("threshold", default)
    return default


def predict_proba(features: np.ndarray, stage: str = "Production") -> Dict[str, Any]:
    model = load_model(stage=stage)
    threshold = load_threshold()
    proba = model.predict(features)
    decision = (proba >= threshold).astype(int)
    return {
        "probability": float(proba[0]),
        "decision": int(decision[0]),
        "threshold": threshold,
    }
