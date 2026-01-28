"""
Page 4: Analyse Bivariée

Scatter plots pour explorer les relations entre deux variables.

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
    page_title="Analyse Bivariée - Prêt à Dépenser",
    page_icon="📊",
    layout="wide",
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.comparison_charts import create_scatter_bivariate
from components.accessibility import add_chart_description, create_page_header
from utils.data_loader import (
    get_client_ids,
    load_clients_data,
    get_feature_names,
    get_feature_label,
    NUMERIC_FEATURES
)

# Page header
create_page_header(
    "Analyse Bivariée",
    "Explorez les relations entre deux variables et positionnez le client."
)

# Sidebar
st.sidebar.header("Configuration")

client_ids = get_client_ids()

if not client_ids:
    st.error("Aucun client disponible.")
    st.stop()

# Client selection
selected_client = st.sidebar.selectbox(
    "Choisir un client",
    options=client_ids,
    format_func=lambda x: f"Client {x}",
    key="bivar_client"
)

# Feature selections
available_features = get_feature_names()

st.sidebar.subheader("Axes du graphique")

feature_x = st.sidebar.selectbox(
    "Variable axe X",
    options=available_features,
    index=0,
    format_func=get_feature_label,
    key="feature_x"
)

feature_y = st.sidebar.selectbox(
    "Variable axe Y",
    options=available_features,
    index=1 if len(available_features) > 1 else 0,
    format_func=get_feature_label,
    key="feature_y"
)

# Color option
color_by_decision = st.sidebar.checkbox(
    "Colorer par décision",
    value=True,
    help="Différencier les clients acceptés et refusés par couleur"
)

# Sample size for performance
sample_size = st.sidebar.slider(
    "Taille de l'échantillon",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Nombre de points affichés (pour les performances)"
)

analyze_clicked = st.sidebar.button(
    "Analyser",
    type="primary",
    use_container_width=True
)

# Main content
if analyze_clicked:
    # Load data
    df = load_clients_data()
    
    if feature_x not in df.columns or feature_y not in df.columns:
        st.error(f"Variables non disponibles dans les données.")
        st.stop()
    
    # Get client values
    if "client_id" in df.columns:
        client_row = df[df["client_id"] == selected_client]
    else:
        client_row = df.loc[[selected_client]]
    
    if client_row.empty:
        st.error(f"Client {selected_client} non trouvé.")
        st.stop()
    
    client_x = client_row[feature_x].values[0]
    client_y = client_row[feature_y].values[0]
    
    if pd.isna(client_x) or pd.isna(client_y):
        st.warning("Valeurs manquantes pour ce client sur les variables sélectionnées.")
        st.stop()
    
    # Sample data for visualization
    plot_df = df[[feature_x, feature_y]].dropna()
    if "target" in df.columns:
        plot_df["target"] = df.loc[plot_df.index, "target"]
    
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=42)
    
    # Create plot
    st.subheader(f"{get_feature_label(feature_x)} vs {get_feature_label(feature_y)}")
    
    fig = create_scatter_bivariate(
        df=plot_df,
        x_feature=feature_x,
        y_feature=feature_y,
        client_x=client_x,
        client_y=client_y,
        color_by="target" if color_by_decision and "target" in plot_df.columns else None
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # WCAG 1.1.1: Text description
    x_label = get_feature_label(feature_x)
    y_label = get_feature_label(feature_y)
    
    # Calculate correlation
    correlation = plot_df[feature_x].corr(plot_df[feature_y])
    
    if correlation > 0.5:
        corr_text = "fortement positive"
    elif correlation > 0.2:
        corr_text = "modérément positive"
    elif correlation > -0.2:
        corr_text = "faible"
    elif correlation > -0.5:
        corr_text = "modérément négative"
    else:
        corr_text = "fortement négative"
    
    add_chart_description(
        f"Le graphique montre la relation entre {x_label} et {y_label}. "
        f"La corrélation est {corr_text} (r = {correlation:.2f}). "
        f"Le client analysé (étoile jaune) a les valeurs {x_label} = {client_x:.2f} et {y_label} = {client_y:.2f}.",
        "scatter plot"
    )
    
    st.divider()
    
    # Client position summary
    st.subheader("Position du client")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # X axis stats
        x_mean = plot_df[feature_x].mean()
        x_pct = (plot_df[feature_x] < client_x).mean() * 100
        
        st.metric(
            label=x_label,
            value=f"{client_x:.2f}",
            delta=f"{x_pct:.0f}ème percentile"
        )
        st.caption(f"Moyenne : {x_mean:.2f}")
    
    with col2:
        # Y axis stats
        y_mean = plot_df[feature_y].mean()
        y_pct = (plot_df[feature_y] < client_y).mean() * 100
        
        st.metric(
            label=y_label,
            value=f"{client_y:.2f}",
            delta=f"{y_pct:.0f}ème percentile"
        )
        st.caption(f"Moyenne : {y_mean:.2f}")
    
    # Correlation info
    st.info(f"**Corrélation** entre {x_label} et {y_label} : **{correlation:.3f}** ({corr_text})")
    
    st.divider()
    
    # Quick analysis by decision group
    if "target" in plot_df.columns:
        st.subheader("Statistiques par groupe")
        
        stats_df = plot_df.groupby("target").agg({
            feature_x: ["mean", "std"],
            feature_y: ["mean", "std"]
        }).round(2)
        
        stats_df.index = ["Acceptés", "Refusés"]
        stats_df.columns = [
            f"{x_label} (moy)", f"{x_label} (écart-type)",
            f"{y_label} (moy)", f"{y_label} (écart-type)"
        ]
        
        st.dataframe(stats_df, use_container_width=True)

else:
    st.info("Sélectionnez un client et deux variables, puis cliquez sur **Analyser**.")
    
    # Show correlation matrix
    st.subheader("Matrice de corrélation")
    
    df = load_clients_data()
    
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    
    if numeric_cols:
        corr_matrix = df[numeric_cols].corr().round(2)
        
        st.dataframe(
            corr_matrix.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
            use_container_width=True
        )
        
        st.caption("""
        **Conseil** : Choisissez deux variables avec une corrélation 
        intéressante (proche de -1 ou +1) pour visualiser la relation.
        """)
