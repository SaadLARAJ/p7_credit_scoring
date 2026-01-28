"""
SHAP visualization components for credit score explainability.

Provides local (client-specific) and global (model-wide) feature importance charts.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from .accessibility import WCAG_COLORS


def create_waterfall_chart(
    shap_values: list[float],
    feature_names: list[str],
    base_value: float,
    max_display: int = 10
) -> go.Figure:
    """
    Create a waterfall chart showing local feature importance (SHAP).
    
    Args:
        shap_values: SHAP values for each feature
        feature_names: Names of features
        base_value: Expected value (baseline)
        max_display: Maximum features to display
        
    Returns:
        Plotly figure object
    """
    # Create DataFrame and sort by absolute importance
    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values
    })
    df["abs_shap"] = df["shap_value"].abs()
    df = df.nlargest(max_display, "abs_shap").sort_values("shap_value")
    
    # Colors based on positive/negative contribution
    colors = [
        WCAG_COLORS["danger"] if v > 0 else WCAG_COLORS["success"]
        for v in df["shap_value"]
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df["feature"],
        x=df["shap_value"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in df["shap_value"]],
        textposition="outside",
        textfont={"size": 12, "color": WCAG_COLORS["text_primary"]},
        hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>"
    ))
    
    # Add baseline annotation
    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="dash",
        line_color=WCAG_COLORS["text_secondary"]
    )
    
    fig.update_layout(
        title={
            "text": "Impact des variables sur la prédiction",
            "font": {"size": 18, "color": WCAG_COLORS["text_primary"]}
        },
        xaxis_title="Impact SHAP (contribution au score)",
        yaxis_title="Variable",
        height=400,
        margin={"l": 150, "r": 50, "t": 60, "b": 50},
        paper_bgcolor=WCAG_COLORS["background"],
        plot_bgcolor=WCAG_COLORS["surface"],
        font={"family": "Arial, sans-serif", "color": WCAG_COLORS["text_primary"]}
    )
    
    return fig


def create_global_importance_chart(
    feature_importances: dict[str, float],
    max_display: int = 10
) -> go.Figure:
    """
    Create a horizontal bar chart showing global feature importance.
    
    Args:
        feature_importances: Dict of feature_name -> importance_value
        max_display: Maximum features to display
        
    Returns:
        Plotly figure object
    """
    # Sort and limit
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:max_display]
    
    features = [f[0] for f in reversed(sorted_features)]
    values = [f[1] for f in reversed(sorted_features)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=values,
        orientation="h",
        marker_color=WCAG_COLORS["info"],
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont={"size": 12, "color": WCAG_COLORS["text_primary"]},
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            "text": "Importance globale des variables",
            "font": {"size": 18, "color": WCAG_COLORS["text_primary"]}
        },
        xaxis_title="Importance moyenne",
        yaxis_title="Variable",
        height=400,
        margin={"l": 150, "r": 50, "t": 60, "b": 50},
        paper_bgcolor=WCAG_COLORS["background"],
        plot_bgcolor=WCAG_COLORS["surface"],
        font={"family": "Arial, sans-serif", "color": WCAG_COLORS["text_primary"]}
    )
    
    return fig


def create_comparison_chart(
    local_shap: dict[str, float],
    global_importance: dict[str, float],
    max_display: int = 8
) -> go.Figure:
    """
    Create a side-by-side comparison of local vs global importance.
    
    Args:
        local_shap: Local SHAP values for current client
        global_importance: Global feature importance
        max_display: Maximum features to display
        
    Returns:
        Plotly figure with subplots
    """
    from plotly.subplots import make_subplots
    
    # Get top features from global importance
    top_features = sorted(
        global_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:max_display]
    feature_names = [f[0] for f in top_features]
    
    # Get local values for same features
    local_values = [local_shap.get(f, 0) for f in feature_names]
    global_values = [global_importance.get(f, 0) for f in feature_names]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Ce client (local)", "Tous les clients (global)"),
        shared_yaxes=True,
        horizontal_spacing=0.1
    )
    
    # Local importance (coloré par signe)
    local_colors = [
        WCAG_COLORS["danger"] if v > 0 else WCAG_COLORS["success"]
        for v in local_values
    ]
    
    fig.add_trace(
        go.Bar(
            y=feature_names,
            x=local_values,
            orientation="h",
            marker_color=local_colors,
            name="Local",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Global importance
    fig.add_trace(
        go.Bar(
            y=feature_names,
            x=global_values,
            orientation="h",
            marker_color=WCAG_COLORS["info"],
            name="Global",
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title={
            "text": "Comparaison importance locale vs globale",
            "font": {"size": 18, "color": WCAG_COLORS["text_primary"]}
        },
        height=450,
        paper_bgcolor=WCAG_COLORS["background"],
        plot_bgcolor=WCAG_COLORS["surface"],
        font={"family": "Arial, sans-serif", "color": WCAG_COLORS["text_primary"]}
    )
    
    return fig
