"""
Generate sample data for Streamlit demo.
Uses synthetic data with correct feature count to avoid preprocessor pickle issues.
"""
import joblib
import numpy as np
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model_final.pkl"
OUTPUT_PATH = ROOT_DIR / "Interface" / "clients_sample.pkl"

def prepare_sample():
    print("Loading model to get expected feature count...")
    model = joblib.load(MODEL_PATH)
    n_features = model.n_features_in_
    print(f"Model expects {n_features} features")
    
    # Generate synthetic client IDs and feature vectors
    # Using random data scaled appropriately for demo purposes
    np.random.seed(42)
    n_clients = 100
    
    data_dict = {}
    for i in range(n_clients):
        client_id = 100001 + i
        # Generate random features (normalized between -1 and 1 for most ML models)
        features = np.random.randn(n_features).tolist()
        data_dict[client_id] = features
    
    print(f"Generated {len(data_dict)} synthetic clients with {n_features} features each")
    joblib.dump(data_dict, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    print("Done!")

if __name__ == "__main__":
    prepare_sample()
