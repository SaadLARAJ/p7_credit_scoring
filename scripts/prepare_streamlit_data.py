import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "joined_clients.csv"
PREPROCESSOR_PATH = ROOT_DIR / "artifacts" / "preprocessor.joblib"
OUTPUT_PATH = ROOT_DIR / "Interface" / "clients_sample.pkl"

def prepare_sample():
    print("Loading data sample...")
    # Load just 500 rows to keep it light
    df = pd.read_csv(DATA_PATH, nrows=500)
    
    # Save IDs
    client_ids = df['client_id'].values
    
    # Prepare X (features)
    # Logic from Src/features/feature_engineering.py: drop target and client_id
    X = df.drop(columns=["target", "client_id"], errors="ignore")
    
    print("Loading preprocessor...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    print("Transforming data...")
    X_transformed = preprocessor.transform(X)
    
    # Handle sparse matrix if necessary (though ColumnTransformer usually returns dense or sparse)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
        
    print(f"Transformed shape: {X_transformed.shape}")
    
    # Create dictionary: ID -> features (list)
    # This is efficient for Streamlit to look up
    data_dict = {}
    for i, client_id in enumerate(client_ids):
        data_dict[int(client_id)] = X_transformed[i].tolist()
        
    print(f"Saving {len(data_dict)} clients to {OUTPUT_PATH}")
    joblib.dump(data_dict, OUTPUT_PATH)
    print("Done!")

if __name__ == "__main__":
    prepare_sample()
