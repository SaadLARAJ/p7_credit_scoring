"""
Page 5: Simulation

Permet de modifier les informations client et recalculer le score.

WCAG Accessibility:
- 1.1.1: Text alternatives
- 1.4.1: Color is not the only means of conveying information
- 1.4.3: Contrast ratio
- 2.4.2: Descriptive page title
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Simulation - Prêt à Dépenser",
    page_icon="📊",
    layout="wide",
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.gauge import create_accessible_gauge
from components.accessibility import (
    format_decision_accessible,
    add_chart_description,
    create_page_header
)
from utils.api_client import get_api_client, test_api_connection
from utils.data_loader import (
    get_client_ids,
    get_client_features,
    load_model_threshold
)

# Page header
create_page_header(
    "Simulation",
    "Modifiez les informations d'un client et voyez l'impact sur le score."
)

# API check
is_connected, api_message = test_api_connection()

if not is_connected:
    st.error(f"{api_message}")
    st.info("Cette fonctionnalité nécessite une connexion à l'API.")
    st.stop()

# Sidebar
st.sidebar.header("Configuration")

client_ids = get_client_ids()

if not client_ids:
    st.error("Aucun client disponible.")
    st.stop()

# Client selection
selected_client = st.sidebar.selectbox(
    "Client de base",
    options=client_ids,
    format_func=lambda x: f"Client {x}",
    key="sim_client"
)

st.sidebar.divider()

# Mode selection
simulation_mode = st.sidebar.radio(
    "Mode de simulation",
    options=["modify", "new"],
    format_func=lambda x: "Modifier un client existant" if x == "modify" else "Créer un nouveau dossier"
)

# Main content
st.warning("""
**Fonctionnalité avancée**

Cette page permet de simuler l'impact de modifications sur le score de crédit.
Les modifications ne sont pas enregistrées et servent uniquement à la simulation.
""")

if simulation_mode == "modify":
    st.subheader("Modifier les caractéristiques")
    
    # Load original features
    original_features = get_client_features(selected_client)
    
    if original_features is None:
        st.error("Impossible de charger les features du client.")
        st.stop()
    
    # Get original prediction
    st.markdown("### Score original")
    
    with st.spinner("Calcul du score original..."):
        try:
            api_client = get_api_client()
            original_result = api_client.predict(selected_client, original_features)
            original_prob = original_result["probability"]
            original_decision = original_result["decision"]
            threshold = original_result["threshold"]
        except Exception as e:
            st.error(f"Erreur API : {e}")
            st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilité originale", f"{original_prob:.1%}")
    with col2:
        icon, text, _ = format_decision_accessible(original_decision, original_prob, threshold)
        if original_decision == 1:
            st.error(f"{icon} {text}")
        else:
            st.success(f"{icon} {text}")
    
    st.divider()
    
    # Feature modification
    st.markdown("### Ajuster les valeurs")
    
    st.info("""
    Les features sont des valeurs numériques transformées par le preprocessing.
    Vous pouvez ajuster les valeurs avec les sliders ci-dessous.
    """)
    
    # Create modifiable features
    modified_features = list(original_features)
    
    # Show first N features for modification
    num_features_to_show = min(10, len(modified_features))
    
    cols = st.columns(2)
    feature_changes = []
    
    for i in range(num_features_to_show):
        original_val = original_features[i]
        col_idx = i % 2
        
        with cols[col_idx]:
            # Determine range based on original value
            if original_val == 0:
                min_val, max_val = -5.0, 5.0
            else:
                min_val = original_val - abs(original_val) * 2
                max_val = original_val + abs(original_val) * 2
            
            new_val = st.slider(
                f"Feature {i+1}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(original_val),
                key=f"feat_{i}"
            )
            
            modified_features[i] = new_val
            
            if new_val != original_val:
                feature_changes.append((i, original_val, new_val))
    
    # Simulate button
    if st.button("Recalculer le score", type="primary"):
        if feature_changes:
            st.markdown("### Modifications appliquées")
            
            for idx, old, new in feature_changes:
                delta = new - old
                st.markdown(f"- Feature {idx+1}: {old:.4f} → {new:.4f} ({delta:+.4f})")
        
        # Get new prediction
        with st.spinner("Calcul du nouveau score..."):
            try:
                new_result = api_client.predict(selected_client, modified_features)
                new_prob = new_result["probability"]
                new_decision = new_result["decision"]
            except Exception as e:
                st.error(f"Erreur API : {e}")
                st.stop()
        
        st.markdown("### Comparaison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Avant**")
            st.metric("Probabilité", f"{original_prob:.1%}")
            icon, text, _ = format_decision_accessible(original_decision, original_prob, threshold)
            if original_decision == 1:
                st.error(f"{icon}")
            else:
                st.success(f"{icon}")
        
        with col2:
            st.markdown("**Après**")
            st.metric(
                "Probabilité",
                f"{new_prob:.1%}",
                delta=f"{(new_prob - original_prob):.1%}"
            )
            icon, text, _ = format_decision_accessible(new_decision, new_prob, threshold)
            if new_decision == 1:
                st.error(f"{icon}")
            else:
                st.success(f"{icon}")
        
        with col3:
            st.markdown("**Impact**")
            delta_prob = new_prob - original_prob
            
            if delta_prob > 0:
                st.warning(f"Risque augmenté de {abs(delta_prob):.1%}")
            elif delta_prob < 0:
                st.success(f"Risque diminué de {abs(delta_prob):.1%}")
            else:
                st.info("Pas de changement")
            
            if original_decision != new_decision:
                st.warning("La décision a changé !")

else:
    # New client mode
    st.subheader("Créer un nouveau dossier")
    
    st.info("""
    Cette fonctionnalité permet de simuler un nouveau dossier client.
    Entrez les caractéristiques du client pour obtenir une estimation du score.
    """)
    
    # Simple form with key features
    with st.form("new_client_form"):
        st.markdown("### Informations client")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=35)
            income = st.number_input("Revenu annuel (€)", min_value=0, max_value=500000, value=45000)
            employment = st.selectbox(
                "Statut d'emploi",
                options=["full_time", "part_time", "self_employed", "unemployed"]
            )
        
        with col2:
            n_transactions = st.number_input("Nombre de transactions", min_value=0, max_value=100, value=5)
            total_spent = st.number_input("Total dépensé (€)", min_value=0, max_value=100000, value=1000)
            avg_interest = st.slider("Taux d'intérêt moyen (%)", 0.0, 30.0, 15.0)
        
        submitted = st.form_submit_button("Calculer le score", type="primary")
    
    if submitted:
        st.warning("""
        **Note importante**
        
        Pour calculer un score réel, les données doivent passer par le pipeline 
        de preprocessing complet. Cette simulation utilise des valeurs simplifiées 
        et ne reflète pas exactement le comportement du modèle.
        """)
        
        # For demo purposes, we use a sample client's features as base
        sample_features = get_client_features(client_ids[0])
        
        if sample_features:
            st.info("Utilisation des features d'un client existant comme base de simulation.")
            
            with st.spinner("Calcul du score..."):
                try:
                    api_client = get_api_client()
                    result = api_client.predict(999999, sample_features)
                    prob = result["probability"]
                    decision = result["decision"]
                    threshold = result["threshold"]
                    
                    fig = create_accessible_gauge(prob, threshold)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    icon, text, _ = format_decision_accessible(decision, prob, threshold)
                    if decision == 1:
                        st.error(f"{icon} {text}")
                    else:
                        st.success(f"{icon} {text}")
                        
                except Exception as e:
                    st.error(f"Erreur : {e}")
