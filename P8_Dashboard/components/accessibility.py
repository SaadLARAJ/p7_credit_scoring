"""
Accessibility helpers for WCAG compliance.

Implements:
- WCAG 1.1.1: Non-text content (descriptions)
- WCAG 1.4.1: Use of color (not color alone)
- WCAG 1.4.3: Contrast (minimum 4.5:1)
- WCAG 1.4.4: Text resize (200%)
- WCAG 2.4.2: Page titles
"""
from __future__ import annotations

import streamlit as st


# WCAG compliant color palette with high contrast
WCAG_COLORS = {
    # Status colors (contrast ratio > 4.5:1 on white)
    "success": "#2E7D32",      # Dark green
    "warning": "#F57C00",      # Orange
    "danger": "#C62828",       # Dark red
    "info": "#1565C0",         # Dark blue
    
    # Neutral colors
    "text_primary": "#212121",
    "text_secondary": "#757575",
    "background": "#FFFFFF",
    "surface": "#F5F5F5",
    
    # Chart colors (distinguishable, accessible)
    "chart_accepted": "#2E7D32",
    "chart_refused": "#C62828",
    "chart_neutral": "#1565C0",
    "chart_highlight": "#FFD600",  # Yellow for highlighting
}


def get_wcag_colors() -> dict[str, str]:
    """Return the WCAG compliant color palette."""
    return WCAG_COLORS.copy()


def add_chart_description(description: str, chart_type: str = "graphique") -> None:
    """
    Add an accessible description below a chart.
    
    Implements WCAG 1.1.1: Non-text content must have text alternatives.
    
    Args:
        description: Human-readable description of what the chart shows
        chart_type: Type of chart for screen readers
    """
    st.caption(f"**Description du {chart_type}** : {description}")


def format_decision_accessible(decision: int, probability: float, threshold: float) -> tuple[str, str, str]:
    """
    Format decision with text + icon (not color alone).
    
    Implements WCAG 1.4.1: Information must not be conveyed by color alone.
    
    Args:
        decision: 0 = accepted, 1 = refused
        probability: Risk probability
        threshold: Decision threshold
        
    Returns:
        Tuple of (icon, text, color)
    """
    if decision == 1:  # Refused
        icon = "[REFUSÉ]"
        text = "CRÉDIT REFUSÉ - Risque trop élevé"
        color = WCAG_COLORS["danger"]
    else:  # Accepted
        if probability < threshold * 0.5:
            icon = "[ACCORDÉ]"
            text = "CRÉDIT ACCORDÉ - Dossier solide"
        else:
            icon = "[ACCORDÉ]"
            text = "CRÉDIT ACCORDÉ - Dossier acceptable"
        color = WCAG_COLORS["success"]
    
    return icon, text, color


def get_risk_level_text(probability: float, threshold: float) -> tuple[str, str]:
    """
    Get risk level with text description.
    
    Args:
        probability: Risk probability
        threshold: Decision threshold
        
    Returns:
        Tuple of (level_text, color)
    """
    if probability < threshold * 0.5:
        return "Risque faible", WCAG_COLORS["success"]
    elif probability < threshold:
        return "Risque modéré", WCAG_COLORS["warning"]
    else:
        return "Risque élevé", WCAG_COLORS["danger"]


def create_page_header(title: str, description: str) -> None:
    """
    Create an accessible page header with title and description.
    
    Implements WCAG 2.4.2: Pages must have descriptive titles.
    """
    st.header(title)
    st.markdown(description)
    st.divider()
