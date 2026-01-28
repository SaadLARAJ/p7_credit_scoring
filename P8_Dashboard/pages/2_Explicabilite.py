"""
Page 2: Explicabilité SHAP

Visualisation de l'importance des features (locale et globale).

WCAG Accessibility:
- 1.1.1: Text alternatives for charts
- 1.4.3: Contrast ratio of at least 4.5:1
- 2.4.2: Descriptive page title
"""
from __future__ import annotations

import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Explicabilité - Prêt à Dépenser",
    page_icon="📊",
    layout="wide",
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.shap_charts import (
    create_waterfall_chart,
    create_global_importance_chart,
    create_comparison_chart
)
from components.accessibility import add_chart_description, create_page_header
from utils.api_client import get_api_client, test_api_connection
from utils.data_loader import (
    get_client_ids,
    get_client_features,
    load_global_shap_importance
)

# Page header
create_page_header(
    "Explicabilité (SHAP)",
    "Comprenez les facteurs qui ont influencé la décision pour ce client."
)

# Sidebar: Client selection
st.sidebar.header("Sélection du client")

client_ids = get_client_ids()

if not client_ids:
    st.error("Aucun client disponible.")
    st.stop()

selected_client = st.sidebar.selectbox(
    "Choisir un client",
    options=client_ids,
    format_func=lambda x: f"Client {x}",
    key="explain_client"
)

# View mode selection
view_mode = st.sidebar.radio(
    "Mode d'affichage",
    options=["Importance locale", "Importance globale", "Comparaison"],
    help="Choisissez ce que vous souhaitez visualiser"
)

# Number of features to display
max_features = st.sidebar.slider(
    "Nombre de variables à afficher",
    min_value=5,
    max_value=20,
    value=10,
    help="Limite le nombre de variables affichées"
)

analyze_clicked = st.sidebar.button(
    "Analyser",
    type="primary",
    use_container_width=True
)

st.sidebar.divider()

# API status
is_connected, _ = test_api_connection()
if is_connected:
    st.sidebar.success("API connectée")
else:
    st.sidebar.error("API déconnectée")

# Main content
if analyze_clicked and selected_client:
    
    features = get_client_features(selected_client)
    
    if features is None:
        st.error(f"Features non disponibles pour le client {selected_client}")
        st.stop()
    
    with st.spinner("Calcul des explications SHAP..."):
        try:
            api_client = get_api_client()
            explain_result = api_client.explain(selected_client, features)
            
            shap_values = explain_result["shap_values"]
            base_value = explain_result["base_value"]
            feature_names = explain_result["feature_names"]
            
        except Exception as e:
            st.error(f"Erreur API : {e}")
            st.stop()
    
    # Load global importance for comparison
    global_importance = load_global_shap_importance()
    
    if view_mode == "Importance locale":
        st.subheader("Facteurs influençant la décision pour ce client")
        
        st.markdown("""
        Ce graphique montre les **variables qui ont le plus influencé** la prédiction pour ce client spécifique.
        
        - Les barres **rouges** indiquent les facteurs qui **augmentent** le risque
        - Les barres **vertes** indiquent les facteurs qui **diminuent** le risque
        """)
        
        fig = create_waterfall_chart(
            shap_values=shap_values,
            feature_names=feature_names,
            base_value=base_value,
            max_display=max_features
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # WCAG 1.1.1: Text description
        top_positive = []
        top_negative = []
        for name, value in zip(feature_names, shap_values):
            if value > 0:
                top_positive.append((name, value))
            else:
                top_negative.append((name, value))
        
        top_positive.sort(key=lambda x: x[1], reverse=True)
        top_negative.sort(key=lambda x: x[1])
        
        description = "Facteurs augmentant le risque : "
        if top_positive[:3]:
            description += ", ".join([f"{n} (+{v:.3f})" for n, v in top_positive[:3]])
        else:
            description += "aucun significatif"
        description += ". Facteurs diminuant le risque : "
        if top_negative[:3]:
            description += ", ".join([f"{n} ({v:.3f})" for n, v in top_negative[:3]])
        else:
            description += "aucun significatif"
        
        add_chart_description(description, "waterfall")
        
    elif view_mode == "Importance globale":
        st.subheader("Variables les plus importantes du modèle")
        
        st.markdown("""
        Ce graphique montre les **variables les plus importantes** pour le modèle 
        en général, sur l'ensemble des clients.
        """)
        
        if global_importance:
            fig = create_global_importance_chart(
                feature_importances=global_importance,
                max_display=max_features
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # WCAG description
            sorted_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            description = "Les 5 variables les plus importantes sont : " + ", ".join(
                [f"{name} ({value:.3f})" for name, value in sorted_features]
            )
            add_chart_description(description, "bar chart")
        else:
            st.warning("Importance globale non disponible.")
            
    else:  # Comparaison
        st.subheader("Comparaison : Ce client vs Modèle global")
        
        st.markdown("""
        Comparez l'importance des variables **pour ce client spécifique** 
        avec leur importance **globale dans le modèle**.
        
        Cela permet d'identifier si ce client a des caractéristiques atypiques.
        """)
        
        # Create comparison data
        local_shap_dict = dict(zip(feature_names, shap_values))
        
        if global_importance:
            fig = create_comparison_chart(
                local_shap=local_shap_dict,
                global_importance=global_importance,
                max_display=max_features
            )
            st.plotly_chart(fig, use_container_width=True)
            
            add_chart_description(
                "Le graphique de gauche montre l'impact des variables pour ce client "
                "(rouge = augmente le risque, vert = diminue). "
                "Le graphique de droite montre l'importance moyenne des variables sur tous les clients.",
                "comparaison"
            )
        else:
            st.warning("Données globales non disponibles pour la comparaison.")
    
    # Display raw values in expander
    with st.expander("Voir les valeurs SHAP brutes"):
        import pandas as pd
        
        df_shap = pd.DataFrame({
            "Variable": feature_names,
            "Valeur SHAP": shap_values
        }).sort_values("Valeur SHAP", key=abs, ascending=False)
        
        st.dataframe(df_shap, use_container_width=True, hide_index=True)
        st.caption(f"Base value (valeur attendue) : {base_value:.4f}")

else:
    st.info("Sélectionnez un client et cliquez sur **Analyser** pour voir les explications.")
    
    # Explanation of SHAP
    with st.expander("Qu'est-ce que SHAP ?"):
        st.markdown("""
        **SHAP** (SHapley Additive exPlanations) est une méthode d'explicabilité 
        qui permet de comprendre comment chaque variable contribue à la prédiction.
        
        ### Interprétation :
        - **Valeur SHAP positive** → la variable **augmente** la probabilité de défaut
        - **Valeur SHAP négative** → la variable **diminue** la probabilité de défaut
        
        ### Types d'importance :
        - **Locale** : Spécifique à un client donné
        - **Globale** : Moyenne sur tous les clients
        
        Cette méthode permet aux chargés de relation client d'expliquer 
        de manière transparente pourquoi un crédit est accordé ou refusé.
        """)
