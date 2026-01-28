"""
Comparison and bivariate analysis chart components.

Provides:
- Histogram with client position marker
- Scatter plot for bivariate analysis
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from .accessibility import WCAG_COLORS


def create_histogram_with_client(
    population_data: pd.Series,
    client_value: float,
    feature_name: str,
    group_name: str = "Tous les clients"
) -> go.Figure:
    """
    Create a histogram showing population distribution with client position.
    
    Args:
        population_data: Series of values for the entire population
        client_value: Value of the current client
        feature_name: Name of the feature being displayed
        group_name: Label for the population group
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Population histogram
    fig.add_trace(go.Histogram(
        x=population_data,
        name=group_name,
        marker_color=WCAG_COLORS["info"],
        opacity=0.7,
        nbinsx=30,
        hovertemplate="Valeur: %{x}<br>Nombre: %{y}<extra></extra>"
    ))
    
    # Client position line
    fig.add_vline(
        x=client_value,
        line_width=4,
        line_dash="solid",
        line_color=WCAG_COLORS["chart_highlight"],
        annotation_text=f"Client: {client_value:.2f}",
        annotation_position="top",
        annotation_font_color=WCAG_COLORS["text_primary"],
        annotation_font_size=14
    )
    
    # Calculate percentile
    percentile = (population_data < client_value).mean() * 100
    mean_value = population_data.mean()
    
    fig.update_layout(
        title={
            "text": f"Distribution de {feature_name}",
            "font": {"size": 18, "color": WCAG_COLORS["text_primary"]}
        },
        xaxis_title=feature_name,
        yaxis_title="Nombre de clients",
        height=400,
        paper_bgcolor=WCAG_COLORS["background"],
        plot_bgcolor=WCAG_COLORS["surface"],
        font={"family": "Arial, sans-serif", "color": WCAG_COLORS["text_primary"]},
        showlegend=True,
        legend={"orientation": "h", "y": -0.2}
    )
    
    # Add annotation with statistics
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Moyenne: {mean_value:.2f}<br>Client au {percentile:.0f}ème percentile",
        showarrow=False,
        font={"size": 12, "color": WCAG_COLORS["text_secondary"]},
        align="left",
        bgcolor=WCAG_COLORS["background"],
        bordercolor=WCAG_COLORS["text_secondary"],
        borderwidth=1
    )
    
    return fig, percentile, mean_value


def create_scatter_bivariate(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    client_x: float,
    client_y: float,
    color_by: str = "target"
) -> go.Figure:
    """
    Create a scatter plot for bivariate analysis with client position.
    
    Args:
        df: DataFrame with population data
        x_feature: Feature for X axis
        y_feature: Feature for Y axis
        client_x: Client's X value
        client_y: Client's Y value
        color_by: Column to use for coloring (default: target)
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Plot by groups if color_by exists
    if color_by in df.columns:
        for group_val, group_name, color in [
            (0, "Acceptés", WCAG_COLORS["chart_accepted"]),
            (1, "Refusés", WCAG_COLORS["chart_refused"])
        ]:
            mask = df[color_by] == group_val
            fig.add_trace(go.Scatter(
                x=df.loc[mask, x_feature],
                y=df.loc[mask, y_feature],
                mode="markers",
                name=group_name,
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.5
                ),
                hovertemplate=f"{x_feature}: %{{x:.2f}}<br>{y_feature}: %{{y:.2f}}<extra>{group_name}</extra>"
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_feature],
            y=df[y_feature],
            mode="markers",
            name="Clients",
            marker=dict(
                color=WCAG_COLORS["info"],
                size=6,
                opacity=0.5
            )
        ))
    
    # Highlight current client with larger, distinct marker
    fig.add_trace(go.Scatter(
        x=[client_x],
        y=[client_y],
        mode="markers+text",
        name="Ce client",
        marker=dict(
            color=WCAG_COLORS["chart_highlight"],
            size=18,
            line=dict(color=WCAG_COLORS["text_primary"], width=3),
            symbol="star"
        ),
        text=["Client"],
        textposition="top center",
        textfont=dict(size=14, color=WCAG_COLORS["text_primary"]),
        hovertemplate=f"<b>Ce client</b><br>{x_feature}: %{{x:.2f}}<br>{y_feature}: %{{y:.2f}}<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            "text": f"Analyse bivariée : {x_feature} vs {y_feature}",
            "font": {"size": 18, "color": WCAG_COLORS["text_primary"]}
        },
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        height=500,
        paper_bgcolor=WCAG_COLORS["background"],
        plot_bgcolor=WCAG_COLORS["surface"],
        font={"family": "Arial, sans-serif", "color": WCAG_COLORS["text_primary"]},
        legend={"orientation": "h", "y": -0.15}
    )
    
    return fig


def get_distribution_stats(
    population_data: pd.Series,
    client_value: float
) -> dict:
    """
    Calculate statistics for client position relative to population.
    
    Args:
        population_data: Population values
        client_value: Client's value
        
    Returns:
        Dictionary with statistics
    """
    return {
        "percentile": (population_data < client_value).mean() * 100,
        "mean": population_data.mean(),
        "median": population_data.median(),
        "std": population_data.std(),
        "min": population_data.min(),
        "max": population_data.max(),
        "client_value": client_value,
        "diff_from_mean": client_value - population_data.mean()
    }
