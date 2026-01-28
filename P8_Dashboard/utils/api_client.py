"""
API Client for communicating with the P7 Credit Scoring API.

The API is deployed on Render: https://p7-credit-scoring-2.onrender.com
"""
from __future__ import annotations

import time
import requests
from typing import Any
import streamlit as st


class APIClient:
    """Client for the Credit Scoring API."""
    
    def __init__(self, base_url: str | None = None):
        """
        Initialize the API client.
        
        Args:
            base_url: API base URL. If None, tries to get from Streamlit secrets.
        """
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            try:
                self.base_url = st.secrets["API_URL"].rstrip("/")
            except (FileNotFoundError, KeyError):
                # Default to the deployed API
                self.base_url = "https://p7-credit-scoring-2.onrender.com"
        
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "P8-Dashboard/1.0"
        }
        # Render free tier can take up to 60s to wake up
        self.timeout = 90
    
    def health_check(self) -> bool:
        """Check if the API is available."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def predict(
        self,
        client_id: int,
        features: list[float]
    ) -> dict[str, Any]:
        """
        Get prediction for a client.
        
        Args:
            client_id: Client identifier
            features: Feature vector
            
        Returns:
            Dictionary with probability, decision, threshold
        """
        payload = {
            "client_id": client_id,
            "features": features
        }
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            headers=self.headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def explain(
        self,
        client_id: int,
        features: list[float]
    ) -> dict[str, Any]:
        """
        Get SHAP explanation for a client prediction.
        
        Args:
            client_id: Client identifier
            features: Feature vector
            
        Returns:
            Dictionary with shap_values, base_value, feature_names
        """
        payload = {
            "client_id": client_id,
            "features": features
        }
        
        response = requests.post(
            f"{self.base_url}/explain",
            json=payload,
            headers=self.headers,
            timeout=120  # SHAP can be slow
        )
        response.raise_for_status()
        
        return response.json()


@st.cache_resource
def get_api_client() -> APIClient:
    """Get a cached API client instance."""
    return APIClient()


def test_api_connection() -> tuple[bool, str]:
    """
    Test the API connection and return status.
    
    Returns:
        Tuple of (is_connected, message)
    """
    client = get_api_client()
    
    try:
        if client.health_check():
            return True, f"Connecté à l'API : {client.base_url}"
        else:
            return False, f"API non disponible : {client.base_url}"
    except Exception as e:
        return False, f"Erreur de connexion : {str(e)}"
