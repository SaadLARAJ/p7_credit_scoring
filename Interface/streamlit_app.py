from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import requests
import shap
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Credit Scoring Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Interface" / "clients_sample.pkl"
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model_final.pkl"
THRESHOLD_PATH = ROOT_DIR / "models" / "optimal_threshold.pkl"

# API Configuration
try:
    API_URL = st.secrets["API_URL"]
except FileNotFoundError:
    API_URL = None
except KeyError:
    API_URL = None

# --- Backend Logic (Cached) ---

@st.cache_resource
def load_data():
    """Load data sample once."""
    if not DATA_PATH.exists():
        st.error(f"Data file not found at {DATA_PATH}")
        return {}
    return joblib.load(DATA_PATH)

@st.cache_resource
def load_local_model():
    """Load model and threshold locally for standalone mode."""
    if not MODEL_PATH.exists():
        return None, 0.5
    model = joblib.load(MODEL_PATH)
    threshold = 0.5
    if THRESHOLD_PATH.exists():
        try:
            threshold = float(joblib.load(THRESHOLD_PATH))
        except:
            pass
    return model, threshold

data_dict = load_data()
local_model, local_threshold = load_local_model()

# Determine mode
USE_API = API_URL is not None and API_URL != ""
MODE = "API" if USE_API else "Local"

# --- UI Layout ---

st.title("🏦 Studio Prêt à Dépenser")
st.markdown(f"**Seuil d'acceptation métier :** `{local_threshold:.3f}`")
if USE_API:
    st.caption(f"🌐 Mode API : {API_URL}")
else:
    st.caption("💻 Mode Local (modèle chargé en mémoire)")

# Sidebar: Client Selection
st.sidebar.header("Dossier Client")

if not data_dict:
    st.error("Aucune donnée client disponible. Veuillez générer 'clients_sample.pkl'.")
    client_ids = []
else:
    client_ids = list(data_dict.keys())

selected_client_id = st.sidebar.selectbox(
    "Choisir un ID Client",
    options=client_ids,
    index=0 if client_ids else None
)

# Main Logic
if selected_client_id:
    features = np.array(data_dict[selected_client_id]).reshape(1, -1)
    
    # Prediction
    if st.sidebar.button("Lancer l'analyse", type="primary"):
        with st.spinner("Analyse du dossier en cours auprès de l'API..."):
            
            # 1. Call API for Prediction
            try:
                payload = {
                    "client_id": selected_client_id,
                    "features": features.tolist()[0]
                }
                # Add headers to mimic a browser and avoid WAF/403 blocks
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                
                response = requests.post(f"{API_URL}/predict", json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                proba = result["probability"]
                decision = result["decision"]
                threshold = result["threshold"]
                
                # 2. Display Result
                st.markdown(f"**Seuil d'acceptation API :** `{threshold:.3f}`")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score de Risque", f"{proba:.2%}")
                with col2:
                    if decision == 1: # 1 means Default/Refused
                        st.error("❌ CRÉDIT REFUSÉ")
                        st.markdown("Risque trop élevé par rapport au seuil.")
                    else:
                        st.success("✅ CRÉDIT ACCORDÉ")
                        st.markdown("Dossier solide.")
                
                # Gauge Bar
                st.progress(min(proba, 1.0))
                
                # 3. SHAP Explanation via API
                st.divider()
                st.subheader("🔍 Explicabilité (SHAP)")
                
                explain_response = requests.post(f"{API_URL}/explain", json=payload, headers=headers)
                explain_response.raise_for_status()
                explain_data = explain_response.json()
                
                # Reconstruct SHAP Explanation object
                # The API returns shap_values, base_value, feature_names, data
                shap_values = np.array(explain_data["shap_values"])
                base_value = explain_data["base_value"]
                feature_names = explain_data["feature_names"]
                data_val = np.array(explain_data["data"])
                
                # Create Explanation object
                # Note: valid for single instance (Waterfall)
                exp_obj = shap.Explanation(
                    values=shap_values,
                    base_values=base_value,
                    data=data_val[0], # data was list of list, take first row
                    feature_names=feature_names
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(exp_obj, show=False)
                st.pyplot(fig)
                
            except requests.exceptions.ConnectionError:
                st.error(f"Impossible de contacter l'API à l'adresse : {API_URL}")
                st.info("Vérifiez que l'API est bien lancée (en local ou sur le cloud).")
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")

# --- Monitoring Section ---
st.divider()
st.markdown("### 📊 Monitoring Data Drift")
st.markdown("Uploadez le rapport Evidently HTML généré par `Src/drift_analysis.py` pour visualiser la dérive des données.")

uploaded_report = st.file_uploader("Déposer rapport Evidently HTML", type=["html"])
if uploaded_report:
    # Display the HTML report inline
    html_content = uploaded_report.read().decode("utf-8")
    st.components.v1.html(html_content, height=800, scrolling=True)
    
    # Also offer download
    st.download_button(
        label="📥 Télécharger le rapport complet",
        data=html_content,
        file_name="drift_report.html",
        mime="text/html"
    )

# --- Footer ---
st.divider()
st.caption("💡 **Note MLflow** : Pour voir l'historique des expériences et métriques, lancez `mlflow ui` en local sur le projet.")
