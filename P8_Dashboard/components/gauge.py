"""
Accessible gauge component for credit score visualization.

Implements WCAG requirements:
- 1.4.1: Color is not the only means of conveying information
- 1.4.3: Contrast ratio of at least 4.5:1
"""
from __future__ import annotations

import plotly.graph_objects as go

from .accessibility import WCAG_COLORS, get_risk_level_text


def create_accessible_gauge(
    value: float,
    threshold: float,
    title: str = "Score de Risque"
) -> go.Figure:
    """
    Create an accessible gauge chart for credit score visualization.
    
    Features:
    - Color zones (green/orange/red)
    - Threshold line clearly visible
    - Text labels for risk levels
    - High contrast colors (WCAG 4.5:1)
    
    Args:
        value: Risk probability (0-1)
        threshold: Decision threshold (0-1)
        title: Gauge title
        
    Returns:
        Plotly figure object
    """
    # Get risk level text and color
    risk_text, risk_color = get_risk_level_text(value, threshold)
    
    # Calculate percentage for display
    value_pct = value * 100
    threshold_pct = threshold * 100
    
    # Create gauge with accessible colors
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value_pct,
        number={
            "suffix": "%",
            "font": {"size": 48, "color": WCAG_COLORS["text_primary"]}
        },
        delta={
            "reference": threshold_pct,
            "increasing": {"color": WCAG_COLORS["danger"]},
            "decreasing": {"color": WCAG_COLORS["success"]},
            "suffix": " pts vs seuil"
        },
        title={
            "text": f"{title}<br><span style='font-size:0.8em;color:{risk_color}'>{risk_text}</span>",
            "font": {"size": 20, "color": WCAG_COLORS["text_primary"]}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 2,
                "tickcolor": WCAG_COLORS["text_primary"],
                "tickfont": {"size": 14, "color": WCAG_COLORS["text_primary"]}
            },
            "bar": {"color": risk_color, "thickness": 0.75},
            "bgcolor": WCAG_COLORS["surface"],
            "borderwidth": 2,
            "bordercolor": WCAG_COLORS["text_secondary"],
            "steps": [
                {
                    "range": [0, threshold_pct * 0.5],
                    "color": "#E8F5E9",  # Light green
                    "name": "Faible"
                },
                {
                    "range": [threshold_pct * 0.5, threshold_pct],
                    "color": "#FFF3E0",  # Light orange
                    "name": "Modéré"
                },
                {
                    "range": [threshold_pct, 100],
                    "color": "#FFEBEE",  # Light red
                    "name": "Élevé"
                },
            ],
            "threshold": {
                "line": {"color": WCAG_COLORS["text_primary"], "width": 4},
                "thickness": 0.8,
                "value": threshold_pct
            },
        }
    ))
    
    # Add threshold annotation
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        text=f"Seuil de décision : {threshold_pct:.1f}%",
        showarrow=False,
        font={"size": 14, "color": WCAG_COLORS["text_secondary"]},
        xref="paper",
        yref="paper"
    )
    
    # Layout for accessibility
    fig.update_layout(
        height=350,
        margin={"t": 80, "b": 60, "l": 40, "r": 40},
        paper_bgcolor=WCAG_COLORS["background"],
        font={"family": "Arial, sans-serif"}
    )
    
    return fig


def create_simple_progress_bar(
    value: float,
    threshold: float
) -> tuple[float, str, str]:
    """
    Create data for a simple progress bar visualization.
    
    Args:
        value: Risk probability (0-1)
        threshold: Decision threshold
        
    Returns:
        Tuple of (progress_value, color, description)
    """
    risk_text, risk_color = get_risk_level_text(value, threshold)
    
    description = f"Score de risque : {value:.1%} (Seuil : {threshold:.1%}) - {risk_text}"
    
    return min(value, 1.0), risk_color, description
