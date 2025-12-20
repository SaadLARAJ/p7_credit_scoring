from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

REPORT_DIR = Path(__file__).resolve().parents[2] / "Monitoring" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets(reference_path: Path, production_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.read_parquet(reference_path)
    production = pd.read_parquet(production_path)
    return reference, production


def build_report(reference: pd.DataFrame, production: pd.DataFrame) -> dict:
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference, current_data=production)
    html_path = REPORT_DIR / "data_drift_report.html"
    json_path = REPORT_DIR / "data_drift_report.json"
    report.save_html(html_path)
    report_json = report.as_dict()
    json_path.write_text(json.dumps(report_json, indent=2))
    print(f"Reports saved to {html_path} and {json_path}")
    return report_json


def alert_if_needed(report_json: dict, threshold: float = 0.3) -> None:
    drift_share = report_json["metrics"][0]["result"]["drift_share"]
    if drift_share >= threshold:
        print(f"ALERT: drift share {drift_share:.2f} >= {threshold}")
        # Integrate webhook (Slack/Teams/Email) here.
    else:
        print(f"Drift share {drift_share:.2f} within acceptable range")


def run_monitoring(reference_path: Path, production_path: Path) -> None:
    reference, production = load_datasets(reference_path, production_path)
    report_json = build_report(reference, production)
    alert_if_needed(report_json)


if __name__ == "__main__":
    REF = Path("artifacts/features/X_valid.parquet")
    PROD = Path("artifacts/features/X_test.parquet")
    run_monitoring(REF, PROD)
