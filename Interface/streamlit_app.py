from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model_final.pkl"
THRESHOLD_PATH = ROOT_DIR / "models" / "optimal_threshold.pkl"
DATA_PATH = ROOT_DIR / "Interface" / "clients_sample.pkl"

# --- Backend Logic (Cached) ---

@st.cache_resource
def load_resources():
    """Load model, threshold and data sample once."""
    if not MODEL_PATH.exists():
        st.error("Model not found! Please ensure 'models/lgbm_model_final.pkl' is in the repo.")
        st.stop()
    
    model = joblib.load(MODEL_PATH)
    
    threshold = 0.5
    if THRESHOLD_PATH.exists():
        try:
            threshold = float(joblib.load(THRESHOLD_PATH))
        except:
            pass
            
    data_dict = {}
    if DATA_PATH.exists():
        data_dict = joblib.load(DATA_PATH)
        
    return model, threshold, data_dict

model, threshold, data_dict = load_resources()

# --- UI Layout ---

st.title("🏦 Studio Prêt à Dépenser")
st.markdown(f"**Seuil d'acceptation métier :** `{threshold:.3f}`")

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
        with st.spinner("Analyse du dossier en cours..."):
            # 1. Probability
            proba = float(model.predict(features)[0])
            decision = proba >= threshold
            
            # 2. Display Result
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score de Risque", f"{proba:.2%}")
            with col2:
                if decision:
                    st.error("❌ CRÉDIT REFUSÉ")
                    st.markdown("Risque trop élevé par rapport au seuil.")
                else:
                    st.success("✅ CRÉDIT ACCORDÉ")
                    st.markdown("Dossier solide.")
            
            # Gauge Bar
            st.progress(min(proba, 1.0))
            
            # 3. SHAP Explanation
            st.divider()
            st.subheader("🔍 Explicabilité (SHAP)")
            st.info("Quelles variables ont le plus impacté cette décision ?")
            
            # Create explainer (TreeExplainer is optimized for LGBM)
            # We use a dummy background if needed, but TreeExplainer handles it well often
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            
            # Handling binary classification shape issues in SHAP
            if isinstance(shap_values, list):
                # LightGBM binary often returns list [class0, class1] or just class1
                vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                vals = shap_values

            # Waterfall plot
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                # Get feature names from model if available
                if hasattr(model, 'feature_name_'):
                    feature_names = model.feature_name_
                else:
                    feature_names = [f"Feature {i}" for i in range(features.shape[1])]
                
                # Create Explanation object for waterfall
                exp_obj = shap.Explanation(
                    values=vals[0],
                    base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    data=features[0],
                    feature_names=feature_names
                )
                shap.plots.waterfall(exp_obj, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Impossible d'afficher le graphique détaillé : {e}")
                st.bar_chart(vals[0])

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
