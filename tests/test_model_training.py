from pathlib import Path

import mlflow
import pytest

from Src.features.feature_engineering import run_feature_engineering
from Src.models.train_model import train
from Src.pipelines.join_datasets import assemble_dataset


@pytest.mark.integration
def test_train_produces_metrics(tmp_path, monkeypatch):
    assemble_dataset()
    run_feature_engineering()
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = tracking_dir.resolve().as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    train()
    assert Path("artifacts/models/gradient_boosting.joblib").exists()
