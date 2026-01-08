"""
Generate sample data for Streamlit demo using correct feature engineering.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model_final.pkl"
DATA_PATH = ROOT_DIR / "data" / "application_train.csv"
OUTPUT_PATH = ROOT_DIR / "Interface" / "clients_sample.pkl"

def prepare_sample():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    feature_names = model.feature_name_
    n_features = len(feature_names)
    print(f"Model expects {n_features} features: {feature_names[:5]}...")
    
    print("Loading real client data...")
    df = pd.read_csv(DATA_PATH, nrows=10000)
    
    # Keep only features the model expects
    available_features = [f for f in feature_names if f in df.columns]
    missing_features = [f for f in feature_names if f not in df.columns]
    
    print(f"Available: {len(available_features)}, Missing: {len(missing_features)}")
    if missing_features:
        print(f"Missing features: {missing_features[:10]}...")
    
    # Select only the features we need, in the right order
    df_features = df[available_features].copy()
    
    # Handle categorical columns - encode them
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            # Simple label encoding
            df_features[col] = pd.factorize(df_features[col])[0]
    
    # Add missing features as zeros
    for feat in missing_features:
        df_features[feat] = 0
    
    # Reorder to match model's expected order
    df_features = df_features[feature_names]
    
    # Fill NaN with median for numeric
    df_features = df_features.fillna(df_features.median())
    
    # Select a mix of clients
    np.random.seed(42)
    good_clients = df[df['TARGET'] == 0].head(200).sample(70, random_state=42).index.tolist()
    bad_clients = df[df['TARGET'] == 1].head(100).sample(30, random_state=42).index.tolist()
    selected = good_clients + bad_clients
    np.random.shuffle(selected)
    
    # Create dictionary
    data_dict = {}
    for idx in selected[:100]:
        client_id = int(df.loc[idx, 'SK_ID_CURR'])
        features = df_features.loc[idx].values.tolist()
        data_dict[client_id] = features
    
    # Verify predictions
    print("\nTesting predictions on sample clients:")
    test_count = 0
    for client_id, features in list(data_dict.items())[:5]:
        X = np.array(features).reshape(1, -1)
        proba = model.predict_proba(X)[0, 1]
        print(f"  Client {client_id}: {proba:.2%}")
        if proba < 0.5:
            test_count += 1
    
    print(f"\n{test_count}/5 test clients would be approved (< 50% risk)")
    
    joblib.dump(data_dict, OUTPUT_PATH)
    print(f"\nSaved {len(data_dict)} clients to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_sample()
