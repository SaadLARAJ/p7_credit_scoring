from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

THRESHOLD_PATH = Path("artifacts/models/threshold.json")


class ClientFeatures(BaseModel):
    client_id: int = Field(..., description="Identifiant client")
    features: list[float] = Field(..., description="Vecteur de features déjà transformé")


app = FastAPI(title="Credit Scoring API", version="1.0.0")


def load_model():
    try:
        return mlflow.pyfunc.load_model("models:/credit_scoring_model/Production")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Model loading failed: {exc}") from exc


def load_threshold(default: float = 0.5) -> float:
    if THRESHOLD_PATH.exists():
        return json.loads(THRESHOLD_PATH.read_text()).get("threshold", default)
    return default


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model_stage": "Production"}


@app.post("/predict")
async def predict(payload: ClientFeatures) -> dict[str, Any]:
    model = load_model()
    threshold = load_threshold()
    array = np.array(payload.features, dtype=float).reshape(1, -1)
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
    model = load_model()
    background = np.zeros((10, len(payload.features)))
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(np.array(payload.features).reshape(1, -1))
    return {
        "client_id": payload.client_id,
        "shap_values": shap_values.tolist(),
    }
