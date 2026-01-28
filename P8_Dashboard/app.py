"""
P8 Dashboard - Main Entry Point

Dashboard interactif pour les chargés de relation client de Prêt à Dépenser.
Permet d'expliquer les décisions d'octroi de crédit de manière transparente.

WCAG Accessibility:
- 2.4.2: Page has a descriptive title
"""
from __future__ import annotations

import streamlit as st

# Page configuration (WCAG 2.4.2: Descriptive page title)
st.set_page_config(
    page_title="Prêt à Dépenser - Dashboard P8",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better accessibility
st.markdown("""
<style>
    /* WCAG 1.4.4: Text can be resized up to 200% without loss of content */
    .stMarkdown {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* High contrast for links */
    a {
        color: #1565C0 !important;
        text-decoration: underline;
    }
    
    /* Better focus indicators */
    button:focus, input:focus, select:focus {
        outline: 3px solid #1976D2;
        outline-offset: 2px;
    }
    
    /* Improve metric readability */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Dashboard Prêt à Dépenser")
st.markdown("""
Bienvenue dans le **Dashboard de Scoring Crédit** (Version P8).

Ce tableau de bord permet aux chargés de relation client de :
""")

# Features
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - **Score Client** : Visualiser le score de crédit et la décision
    - **Explicabilité** : Comprendre les facteurs de la décision (SHAP)
    """)

with col2:
    st.markdown("""
    - **Comparaison** : Comparer un client à la population
    - **Analyse Bivariée** : Explorer les relations entre variables
    """)

st.divider()

# API Status
st.subheader("État de la connexion")

from utils.api_client import test_api_connection

is_connected, message = test_api_connection()

if is_connected:
    st.success(message)
else:
    st.error(message)
    st.info("""
    **L'API n'est pas joignable.**
    
    Cela peut arriver si l'API sur Render est en veille (cold start).
    Veuillez patienter quelques secondes et réessayer.
    
    Vous pouvez également vérifier l'état de l'API ici :
    [API Status](https://p7-credit-scoring-2.onrender.com/docs)
    """)

st.divider()

# Navigation
st.subheader("Navigation")

st.markdown("""
Utilisez le **menu latéral** pour naviguer entre les pages :

| Page | Description |
|------|-------------|
| Score Client | Visualiser le score et la décision |
| Explicabilité | Comprendre les facteurs (SHAP) |
| Comparaison | Comparer à la population |
| Analyse Bivariée | Explorer les relations |
| Simulation | Modifier un dossier |
""")

st.divider()

# Footer with accessibility note
st.caption("""
**Accessibilité** : Ce dashboard respecte les critères WCAG 2.1 niveau AA pour 
garantir l'accessibilité aux personnes en situation de handicap.
Les graphiques incluent des descriptions textuelles et n'utilisent pas uniquement la couleur 
pour transmettre l'information.
""")

st.caption("""
**Données** : Les données utilisées proviennent du jeu de données Home Credit Default Risk.
Le modèle de scoring est basé sur LightGBM avec optimisation du seuil de décision pour le coût métier.
""")
