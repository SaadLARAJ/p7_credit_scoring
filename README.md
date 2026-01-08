# Credit Scoring – "Prêt à Dépenser"

Plateforme de scoring crédit end-to-end avec pipeline ML, API REST, interface Streamlit et monitoring de dérive.

## Démarrage rapide

```bash
# Environnement
python -m venv env && .\env\Scripts\Activate.ps1
pip install -r requirements.txt

# MLflow (terminal séparé)
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns.db

# Pipeline complet
python Src/pipelines/join_datasets.py
python Src/features/feature_engineering.py
python Src/models/train_model.py

# Serving
uvicorn Api.app.main:app --reload --port 8000
streamlit run Interface/streamlit_app.py
```

## Architecture

```
├── Api/app/main.py           # FastAPI : /health, /predict, /explain
├── Interface/streamlit_app.py # Dashboard analyste
├── Src/
│   ├── pipelines/            # Jointure multi-tables
│   ├── features/             # Feature engineering + split
│   ├── models/               # Entraînement + score métier
│   └── monitoring/           # Rapports Evidently
├── tests/                    # Pytest (pipeline, API, modèle)
├── .github/workflows/ci.yml  # CI/CD automatisé
└── docker/                   # Dockerfiles API + Streamlit
```

## Score métier

Le modèle optimise un coût asymétrique :
- **Faux Négatif** (prêt accordé à un mauvais payeur) : coût ×10
- **Faux Positif** (prêt refusé à un bon payeur) : coût ×1

Le seuil de décision est ajusté automatiquement pour minimiser ce coût global.

## API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Status du service |
| `/predict` | POST | Probabilité de défaut + décision |
| `/explain` | POST | Valeurs SHAP locales |

**Exemple :**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"client_id": 100001, "features": [0.1, 0.2, ...]}'
```

## Tests

```bash
pytest --maxfail=1 -q
```

Couverture : pipeline de données, entraînement (marqué `@pytest.mark.integration`), endpoints API.

## Monitoring

Génération du rapport de dérive :
```bash
python Src/monitoring/drift_monitor.py
```

Seuil d'alerte : drift > 30% des features.

## Déploiement

Les Dockerfiles sont prêts pour Render, Railway ou Azure Container Apps :
```bash
docker build -f docker/Dockerfile.api -t credit-scoring-api .
docker build -f docker/Dockerfile.streamlit -t credit-scoring-app .
```

## Liens

- **Streamlit Cloud** : [lien à ajouter après déploiement]

