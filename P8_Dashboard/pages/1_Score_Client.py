"""
Page 1: Score Client

Visualisation du score de crédit avec jauge accessible et décision.

WCAG Accessibility:
- 1.1.1: Text alternatives for charts
- 1.4.1: Color is not the only means of conveying information
- 1.4.3: Contrast ratio of at least 4.5:1
- 2.4.2: Descriptive page title
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Score Client - Prêt à Dépenser",
    page_icon="📊",
    layout="wide",
)

# Imports after page config
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.gauge import create_accessible_gauge
from components.accessibility import (
    format_decision_accessible,
    add_chart_description,
    create_page_header,
    WCAG_COLORS
)
from utils.api_client import get_api_client, test_api_connection
from utils.data_loader import (
    get_client_ids,
    get_client_info,
    get_client_features,
    load_model_threshold,
    FEATURE_LABELS
)

# Page header
create_page_header(
    "Score Client",
    "Visualisez le score de crédit d'un client et la décision d'octroi."
)

# Sidebar: Client selection
st.sidebar.header("Sélection du client")

client_ids = get_client_ids()

if not client_ids:
    st.error("Aucun client disponible. Vérifiez que les données sont chargées.")
    st.stop()

selected_client = st.sidebar.selectbox(
    "Choisir un client",
    options=client_ids,
    format_func=lambda x: f"Client {x}",
    help="Sélectionnez un client pour voir son score de crédit"
)

# Action button
analyze_clicked = st.sidebar.button(
    "Analyser le dossier",
    type="primary",
    use_container_width=True
)

st.sidebar.divider()

# API status in sidebar
is_connected, api_message = test_api_connection()
if is_connected:
    st.sidebar.success("API connectée")
else:
    st.sidebar.error("API déconnectée")

# Load default threshold
default_threshold = load_model_threshold()
st.sidebar.info(f"Seuil de décision : {default_threshold:.2%}")

# Main content
if analyze_clicked and selected_client:
    
    # Get client features for API call
    features = get_client_features(selected_client)
    
    if features is None:
        st.error(f"Impossible de charger les features du client {selected_client}")
        st.info("Les features du client doivent être disponibles dans Interface/clients_sample.pkl")
        st.stop()
    
    # Call API
    with st.spinner("Analyse du dossier en cours..."):
        try:
            api_client = get_api_client()
            result = api_client.predict(selected_client, features)
            
            probability = result["probability"]
            decision = result["decision"]
            threshold = result["threshold"]
            
        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")
            st.info("Vérifiez que l'API est disponible et réessayez.")
            st.stop()
    
    # Display results
    st.subheader("Résultat de l'analyse")
    
    # Two columns: Gauge and Decision
    col_gauge, col_decision = st.columns([2, 1])
    
    with col_gauge:
        # Accessible gauge
        gauge_fig = create_accessible_gauge(probability, threshold)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # WCAG 1.1.1: Text alternative
        add_chart_description(
            f"Le score de risque est de {probability:.1%}. "
            f"Le seuil de décision est fixé à {threshold:.1%}. "
            f"{'Le score est supérieur au seuil, le crédit est refusé.' if decision == 1 else 'Le score est inférieur au seuil, le crédit est accordé.'}",
            "jauge"
        )
    
    with col_decision:
        st.markdown("### Décision")
        
        # WCAG 1.4.1: Use text + icon + color
        icon, text, color = format_decision_accessible(decision, probability, threshold)
        
        if decision == 1:
            st.error(f"{icon} {text}")
        else:
            st.success(f"{icon} {text}")
        
        # Metrics
        st.metric(
            label="Score de risque",
            value=f"{probability:.1%}",
            delta=f"{(probability - threshold):.1%} vs seuil",
            delta_color="inverse"
        )
        
        st.metric(
            label="Seuil de décision",
            value=f"{threshold:.1%}"
        )
    
    st.divider()
    
    # Client information
    st.subheader("Informations du client")
    
    client_info = get_client_info(selected_client)
    
    if client_info:
        # Display in a nice table format
        col1, col2 = st.columns(2)
        
        info_items = list(client_info.items())
        mid_point = len(info_items) // 2
        
        with col1:
            for key, value in info_items[:mid_point]:
                if key != "Défaut de paiement":  # Don't show target
                    if isinstance(value, float):
                        st.markdown(f"**{key}:** {value:,.2f}")
                    else:
                        st.markdown(f"**{key}:** {value}")
        
        with col2:
            for key, value in info_items[mid_point:]:
                if key != "Défaut de paiement":
                    if isinstance(value, float):
                        st.markdown(f"**{key}:** {value:,.2f}")
                    else:
                        st.markdown(f"**{key}:** {value}")
    else:
        st.info("Informations descriptives non disponibles pour ce client.")

else:
    # Initial state
    st.info("Sélectionnez un client dans la barre latérale et cliquez sur **Analyser le dossier**.")
    
    # Show available clients count
    st.subheader("Clients disponibles")
    st.write(f"{len(client_ids)} clients avec features disponibles pour l'analyse.")
