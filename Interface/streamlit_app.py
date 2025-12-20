from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="Credit Scoring Studio", layout="wide")
st.title("Studio scoring client")

api_url = st.sidebar.text_input("URL de l'API", "http://localhost:8000")

with st.form("prediction_form"):
    client_id = st.number_input("Client ID", value=100001)
    features_raw = st.text_area("Vecteur de features (JSON)", value="[0.1, 0.2, 0.5]")
    submitted = st.form_submit_button("Lancer la prédiction")

if submitted:
    try:
        features: Any = json.loads(features_raw)
    except json.JSONDecodeError as exc:
        st.error(f"JSON invalide: {exc}")
    else:
        response = requests.post(
            f"{api_url}/predict",
            json={"client_id": client_id, "features": features},
            timeout=30,
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Probabilité de défaut: {result['probability']:.2%}")
            st.metric("Décision", "Refus" if result["decision"] else "Acceptation")
            st.caption(f"Seuil métier appliqué: {result['threshold']:.2f}")
            if st.checkbox("Voir l'explication SHAP"):
                explanation = requests.post(
                    f"{api_url}/explain",
                    json={"client_id": client_id, "features": features},
                    timeout=30,
                )
                if explanation.status_code == 200:
                    st.json(explanation.json())
                else:
                    st.error(f"Impossible de récupérer les explications ({explanation.status_code}).")
        else:
            st.error(f"Erreur API {response.status_code}: {response.text}")

st.markdown("### Historique de monitoring")
uploaded_report = st.file_uploader("Déposer le rapport Evidently HTML", type=["html"])
if uploaded_report:
    st.download_button(
        "Télécharger le rapport",
        data=uploaded_report,
        file_name="data_drift_report.html",
    )
