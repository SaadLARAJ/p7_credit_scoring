"""
Data loading utilities for the P8 Dashboard.

Loads client data from the P7 project artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import numpy as np
import streamlit as st


# Path configuration
DASHBOARD_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DASHBOARD_DATA_DIR = DASHBOARD_DIR / "data"  # For Streamlit Cloud deployment
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
INTERFACE_DIR = PROJECT_ROOT / "Interface"


# Features that can be displayed to the user (descriptive)
DESCRIPTIVE_FEATURES = [
    "gender", "age", "income", "employment_status", "country",
    "n_transactions", "total_spent", "avg_ticket", "days_since_last",
    "avg_interest_rate", "max_tenor"
]

# Features available for comparison charts
NUMERIC_FEATURES = [
    "age", "income", "n_transactions", "total_spent", 
    "avg_ticket", "days_since_last", "avg_interest_rate", "max_tenor"
]

# Human-readable labels for features
FEATURE_LABELS = {
    "gender": "Genre",
    "age": "Âge",
    "income": "Revenu annuel (€)",
    "employment_status": "Statut d'emploi",
    "country": "Pays",
    "n_transactions": "Nombre de transactions",
    "total_spent": "Total dépensé (€)",
    "avg_ticket": "Panier moyen (€)",
    "days_since_last": "Jours depuis dernière transaction",
    "avg_interest_rate": "Taux d'intérêt moyen (%)",
    "max_tenor": "Durée maximale (mois)",
    "target": "Défaut de paiement"
}


@st.cache_data
def load_feature_vectors() -> dict[int, list[float]]:
    """
    Load the preprocessed feature vectors for API calls.
    
    Returns:
        Dictionary mapping client_id to feature vector
    """
    sample_path = INTERFACE_DIR / "clients_sample.pkl"
    if sample_path.exists():
        return joblib.load(sample_path)
    
    return {}


@st.cache_data
def load_clients_data() -> pd.DataFrame:
    """
    Load the joined clients dataset.
    
    Returns:
        DataFrame with all client data including descriptive features
    """
    # Try P8_Dashboard/data first (for Streamlit Cloud deployment)
    dashboard_path = DASHBOARD_DATA_DIR / "joined_clients.csv"
    if dashboard_path.exists():
        df = pd.read_csv(dashboard_path)
        return df
    
    # Try project root data folder (local development)
    joined_path = DATA_DIR / "joined_clients.csv"
    if joined_path.exists():
        df = pd.read_csv(joined_path)
        return df
    
    # Fallback to clients_sample in Interface
    sample_path = INTERFACE_DIR / "clients_sample.pkl"
    if sample_path.exists():
        data_dict = joblib.load(sample_path)
        # This is just features, no descriptive columns
        return pd.DataFrame.from_dict(data_dict, orient="index")
    
    raise FileNotFoundError(
        "No client data found. Expected: P8_Dashboard/data/joined_clients.csv or data/joined_clients.csv"
    )


@st.cache_resource
def load_model_threshold() -> float:
    """Load the optimal decision threshold from P7."""
    threshold_path = MODELS_DIR / "optimal_threshold.pkl"
    if threshold_path.exists():
        return float(joblib.load(threshold_path))
    return 0.5  # Default


@st.cache_resource  
def load_global_shap_importance() -> dict[str, float]:
    """
    Load or compute global SHAP importance.
    
    Returns:
        Dictionary of feature_name -> mean absolute SHAP value
    """
    # Try to load pre-computed global importance
    global_path = MODELS_DIR / "global_shap_importance.pkl"
    if global_path.exists():
        return joblib.load(global_path)
    
    # Fallback: use model feature importances
    model_path = MODELS_DIR / "lgbm_model_final.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        if hasattr(model, "feature_importances_"):
            # Get feature names if available
            if hasattr(model, "feature_name_"):
                names = model.feature_name_
            elif hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
            else:
                names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            
            return dict(zip(names, model.feature_importances_.tolist()))
    
    return {}


def get_client_ids() -> list[int]:
    """
    Get list of available client IDs from the feature vectors pickle.
    These are the clients that have feature vectors for API prediction.
    """
    # Use IDs from the pickle file (these are the ones with feature vectors)
    feature_vectors = load_feature_vectors()
    if feature_vectors:
        return sorted(list(feature_vectors.keys()))
    
    # Fallback to CSV
    df = load_clients_data()
    if "client_id" in df.columns:
        return df["client_id"].tolist()
    return df.index.tolist()


def get_client_info(client_id: int) -> dict[str, Any]:
    """
    Get descriptive information for a client.
    
    Args:
        client_id: Client identifier
        
    Returns:
        Dictionary with client's descriptive features
    """
    # For now, return feature vector info since descriptive data may not match
    feature_vectors = load_feature_vectors()
    if client_id in feature_vectors:
        features = feature_vectors[client_id]
        return {
            "Client ID": client_id,
            "Nombre de features": len(features),
            "Note": "Données descriptives non disponibles pour ce client"
        }
    
    # Try from CSV
    df = load_clients_data()
    
    if "client_id" in df.columns:
        client_row = df[df["client_id"] == client_id]
    else:
        try:
            client_row = df.loc[[client_id]]
        except KeyError:
            return {}
    
    if client_row.empty:
        return {}
    
    info = client_row.iloc[0].to_dict()
    
    # Add human-readable labels
    labeled_info = {}
    for key, value in info.items():
        label = FEATURE_LABELS.get(key, key)
        labeled_info[label] = value
    
    return labeled_info


def get_client_features(client_id: int) -> list[float] | None:
    """
    Get the feature vector for API prediction.
    
    Args:
        client_id: Client identifier
        
    Returns:
        Feature vector as list of floats, or None if not found
    """
    feature_vectors = load_feature_vectors()
    return feature_vectors.get(client_id)


def get_feature_names() -> list[str]:
    """Get list of feature names for charts."""
    return NUMERIC_FEATURES.copy()


def get_feature_label(feature_name: str) -> str:
    """Get human-readable label for a feature."""
    return FEATURE_LABELS.get(feature_name, feature_name)


def get_population_data(feature: str, group: str = "all") -> pd.Series:
    """
    Get population data for a specific feature.
    
    Args:
        feature: Feature name
        group: Filter group ("all", "accepted", "refused")
        
    Returns:
        Series of feature values
    """
    df = load_clients_data()
    
    if feature not in df.columns:
        return pd.Series([])
    
    if group == "accepted" and "target" in df.columns:
        df = df[df["target"] == 0]
    elif group == "refused" and "target" in df.columns:
        df = df[df["target"] == 1]
    
    return df[feature].dropna()
