"""
Generate Data Drift Report using Evidently AI.
Uses application_train.csv vs application_test.csv (or joined_clients.csv samples).
"""
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_PATH = ROOT_DIR / "drift_report.html"

def generate_drift_report():
    print("Loading reference data (train sample)...")
    # Use joined_clients.csv if available, otherwise application_train.csv
    data_path = DATA_DIR / "joined_clients.csv"
    if not data_path.exists():
        data_path = DATA_DIR / "application_train.csv"
    
    df = pd.read_csv(data_path, nrows=2000)
    print(f"Loaded {len(df)} rows from {data_path.name}")
    
    # Split into reference (first half) and current (second half) to simulate drift
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()
    
    # Select only numeric columns to avoid issues
    numeric_cols = reference.select_dtypes(include=['number']).columns.tolist()
    # Limit to first 30 columns to keep report readable
    cols_to_use = numeric_cols[:30]
    
    reference = reference[cols_to_use]
    current = current[cols_to_use]
    
    print(f"Analyzing drift on {len(cols_to_use)} numeric features...")
    
    # Generate report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    
    # Save
    report.save_html(OUTPUT_PATH)
    print(f"Drift report saved to: {OUTPUT_PATH}")
    print("Done!")

if __name__ == "__main__":
    generate_drift_report()
