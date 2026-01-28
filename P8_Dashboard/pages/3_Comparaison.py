"""
Page 3: Comparaison Client vs Population

Histogrammes de distribution avec position du client.

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
    page_title="Comparaison - Prêt à Dépenser",
    page_icon="📊",
    layout="wide",
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.comparison_charts import (
    create_histogram_with_client,
    get_distribution_stats
)
from components.accessibility import add_chart_description, create_page_header
from utils.data_loader import (
    get_client_ids,
    get_client_info,
    load_clients_data,
    get_feature_names,
    get_feature_label,
    get_population_data,
    NUMERIC_FEATURES
)

# Page header
create_page_header(
    "Comparaison Client vs Population",
    "Visualisez comment ce client se situe par rapport à l'ensemble des clients."
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
    key="compare_client"
)

# Feature selection
available_features = get_feature_names()
selected_feature = st.sidebar.selectbox(
    "Variable à analyser",
    options=available_features,
    format_func=get_feature_label,
    help="Choisissez la variable à comparer"
)

# Group filter
comparison_group = st.sidebar.radio(
    "Comparer avec",
    options=["all", "accepted", "refused"],
    format_func=lambda x: {
        "all": "Tous les clients",
        "accepted": "Clients acceptés (crédit accordé)",
        "refused": "Clients refusés"
    }[x],
    help="Filtrez la population de comparaison"
)

analyze_clicked = st.sidebar.button(
    "Comparer",
    type="primary",
    use_container_width=True
)

# Main content
if analyze_clicked:
    # Load data
    df = load_clients_data()
    
    # Get client info
    client_info = get_client_info(selected_client)
    
    if not client_info:
        st.error(f"Client {selected_client} non trouvé.")
        st.stop()
    
    # Get client value for selected feature
    feature_label = get_feature_label(selected_feature)
    client_value = client_info.get(feature_label)
    
    if client_value is None:
        # Try raw feature name
        if "client_id" in df.columns:
            client_row = df[df["client_id"] == selected_client]
        else:
            client_row = df.loc[[selected_client]]
        
        if not client_row.empty and selected_feature in client_row.columns:
            client_value = client_row[selected_feature].values[0]
    
    if client_value is None or pd.isna(client_value):
        st.warning(f"Valeur non disponible pour {feature_label} pour ce client.")
        st.stop()
    
    # Get population data
    population_data = get_population_data(selected_feature, comparison_group)
    
    if population_data.empty:
        st.warning(f"Pas de données disponibles pour {feature_label}.")
        st.stop()
    
    # Display histogram
    st.subheader(f"Distribution : {feature_label}")
    
    group_label = {
        "all": "Tous les clients",
        "accepted": "Clients acceptés",
        "refused": "Clients refusés"
    }[comparison_group]
    
    fig, percentile, mean_value = create_histogram_with_client(
        population_data=population_data,
        client_value=client_value,
        feature_name=feature_label,
        group_name=group_label
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # WCAG 1.1.1: Text description
    if client_value > mean_value:
        position_text = "supérieur à"
        diff_text = f"+{client_value - mean_value:.2f}"
    else:
        position_text = "inférieur à"
        diff_text = f"{client_value - mean_value:.2f}"
    
    add_chart_description(
        f"Le client a une valeur de {client_value:.2f} pour {feature_label}, "
        f"ce qui le place au {percentile:.0f}ème percentile. "
        f"Cette valeur est {position_text} la moyenne de {mean_value:.2f} ({diff_text}).",
        "histogramme"
    )
    
    st.divider()
    
    # Statistics panel
    st.subheader("Statistiques")
    
    stats = get_distribution_stats(population_data, client_value)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Valeur du client",
            value=f"{stats['client_value']:.2f}"
        )
    
    with col2:
        st.metric(
            label="Percentile",
            value=f"{stats['percentile']:.0f}ème",
            help="Position du client dans la distribution"
        )
    
    with col3:
        st.metric(
            label="Moyenne population",
            value=f"{stats['mean']:.2f}",
            delta=f"{stats['diff_from_mean']:.2f}",
            delta_color="off"
        )
    
    with col4:
        st.metric(
            label="Médiane population",
            value=f"{stats['median']:.2f}"
        )
    
    # Additional stats in expander
    with st.expander("Statistiques détaillées"):
        st.markdown(f"""
        | Statistique | Valeur |
        |-------------|--------|
        | Min | {stats['min']:.2f} |
        | Max | {stats['max']:.2f} |
        | Écart-type | {stats['std']:.2f} |
        | Médiane | {stats['median']:.2f} |
        | Moyenne | {stats['mean']:.2f} |
        """)
    
    st.divider()
    
    # Quick comparison of multiple features
    st.subheader("Comparer d'autres variables")
    
    # Select multiple features
    other_features = st.multiselect(
        "Sélectionner d'autres variables",
        options=[f for f in available_features if f != selected_feature],
        format_func=get_feature_label,
        max_selections=4
    )
    
    if other_features:
        cols = st.columns(len(other_features))
        
        for i, feature in enumerate(other_features):
            with cols[i]:
                feat_label = get_feature_label(feature)
                pop_data = get_population_data(feature, comparison_group)
                
                # Get client value
                if "client_id" in df.columns:
                    client_row = df[df["client_id"] == selected_client]
                else:
                    client_row = df.loc[[selected_client]]
                
                if not client_row.empty and feature in client_row.columns:
                    val = client_row[feature].values[0]
                    if not pd.isna(val) and not pop_data.empty:
                        pct = (pop_data < val).mean() * 100
                        st.metric(
                            label=feat_label,
                            value=f"{val:.2f}",
                            delta=f"{pct:.0f}ème pct"
                        )

else:
    st.info("Sélectionnez un client et une variable, puis cliquez sur **Comparer**.")
    
    # Show available features
    st.subheader("Variables disponibles pour comparaison")
    
    features_df = pd.DataFrame({
        "Variable": available_features,
        "Description": [get_feature_label(f) for f in available_features]
    })
    
    st.dataframe(features_df, use_container_width=True, hide_index=True)
