"""
Generate realistic sample data for Streamlit demo.
Uses actual Home Credit data to create representative client samples.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model_final.pkl"
DATA_PATH = ROOT_DIR / "data" / "application_train.csv"
OUTPUT_PATH = ROOT_DIR / "Interface" / "clients_sample.pkl"

def prepare_sample():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    n_features = model.n_features_in_
    print(f"Model expects {n_features} features")
    
    print("Loading real client data...")
    # Load a sample of real data
    df = pd.read_csv(DATA_PATH, nrows=5000)
    
    # Get feature columns (exclude ID and target)
    exclude_cols = ['SK_ID_CURR', 'TARGET']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Select numeric columns only and handle missing values
    df_numeric = df[feature_cols].select_dtypes(include=[np.number])
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    # If we have more features than needed, select first n_features
    # If fewer, pad with zeros (shouldn't happen with real data)
    if len(df_numeric.columns) >= n_features:
        df_features = df_numeric.iloc[:, :n_features]
    else:
        # Pad with zeros if needed
        padding = np.zeros((len(df_numeric), n_features - len(df_numeric.columns)))
        df_features = pd.concat([
            df_numeric, 
            pd.DataFrame(padding, columns=[f"pad_{i}" for i in range(padding.shape[1])])
        ], axis=1)
    
    # Create sample dictionary with mix of good and bad clients
    np.random.seed(42)
    
    # Get indices for both classes
    good_clients = df[df['TARGET'] == 0].index[:70].tolist()  # 70 good
    bad_clients = df[df['TARGET'] == 1].index[:30].tolist()   # 30 bad
    selected_indices = good_clients + bad_clients
    np.random.shuffle(selected_indices)
    
    data_dict = {}
    for i, idx in enumerate(selected_indices[:100]):
        client_id = int(df.loc[idx, 'SK_ID_CURR'])
        features = df_features.loc[idx].values.tolist()
        data_dict[client_id] = features
    
    print(f"Generated {len(data_dict)} real clients with {n_features} features each")
    print(f"Mix: ~70% good clients, ~30% risky clients")
    joblib.dump(data_dict, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    print("Done!")

if __name__ == "__main__":
    prepare_sample()
