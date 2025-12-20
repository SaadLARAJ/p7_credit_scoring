# Dossier de tests

Déposez ici les suites Pytest. Couverture minimale attendue :
1. **Tests données** – vérifier schémas, ratio de classes, seuils de valeurs manquantes après export du Notebook 01.
2. **Tests modèle** – prédictions déterministes sur un petit jeu de fixtures, s’assurer que le seuil métier est bien enregistré avec le modèle.
3. **Tests API** – FastAPI TestClient pour `/health` et `/predict` (schémas + logique coût).
4. **Smoke tests Streamlit** (optionnel) – `pytest -k streamlit` avec `pytest-streamlit` ou snapshots.

Commande CI recommandée :
```bash
pytest --maxfail=1 --disable-warnings -q
```
Ajoutez la couverture (`pytest-cov`) ultérieurement. Mettez à jour ce document quand de nouveaux tests apparaissent.
