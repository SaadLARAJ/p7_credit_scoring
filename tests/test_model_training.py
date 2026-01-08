"""
Integration tests for model training pipeline.
"""
from pathlib import Path

import mlflow
import pytest

from Src.features.feature_engineering import run_feature_engineering
from Src.models.train_model import train
from Src.pipelines.join_datasets import assemble_dataset


@pytest.mark.integration
def test_train_produces_model(tmp_path, monkeypatch):
    """Test that training pipeline produces a valid LightGBM model."""
    # Setup
    assemble_dataset()
    run_feature_engineering()
    
    # Configure MLflow to use temp directory
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = tracking_dir.resolve().as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    
    # Run training
    train()
    
    # Verify outputs
    assert Path("models/lgbm_model_final.pkl").exists(), "Model file not created"
    assert Path("models/optimal_threshold.pkl").exists(), "Threshold file not created"
