# Plateforme complète de scoring client & architecture MLOps

Ce socle fournit tout le nécessaire pour construire un scoring client end-to-end : préparation multi-tables, feature engineering, entraînement suivi par MLflow, API FastAPI, interface Streamlit, monitoring Evidently, CI/CD GitHub Actions et conteneurisation Docker. Tout le code est prêt à l’emploi : il suffit d’exécuter les scripts dans l’ordre décrit ci-dessous.

---

## 1. Architecture générale

- **Collecte & préparation** : scripts `Src/pipelines` joignent les tables clients/transactions/produits (extraits fictifs sous `data/samples/` pour tester rapidement).
- **Feature engineering** : `Src/features` prépare les matrices d’apprentissage, gère les déséquilibres et sauvegarde les splits avec les poids d’échantillons.
- **Modélisation & MLOps** : `Src/models` orchestre GridSearchCV, MLflow tracking/registry, métriques métier (AUC & coût), SHAP global/local, seuil métier et export d’artefacts.
- **Serving & Interface** : `Api/app/main.py` (FastAPI) sert les prédictions, `Interface/streamlit_app.py` offre une UI analyste connectée à l’API.
- **Monitoring & alertes** : `Src/monitoring/drift_monitor.py` automatise les rapports Evidently, `Monitoring/` stocke rapports + logs.
- **Qualité & automatisation** : tests Pytest (`tests/`), pipeline `.github/workflows/ci.yml`, Dockerfiles (`docker/`).

### Arborescence de référence

```
P7_credit_scoring/
├── Api/
│   └── app/main.py                    # API FastAPI
├── Interface/
│   └── streamlit_app.py               # UI Streamlit
├── Src/
│   ├── pipelines/join_datasets.py     # Jointure multi-tables
│   ├── features/feature_engineering.py# Feature store + splitting
│   ├── models/
│   │   ├── custom_score.py            # Score métier
│   │   └── train_model.py             # GridSearch + MLflow
│   ├── inference/predict.py           # Chargement registry + scoring
│   └── monitoring/drift_monitor.py    # Rapports Evidently + alertes
├── Monitoring/README.md               # Consignes reporting & alertes
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.streamlit
├── configs/
│   ├── mlflow.yaml
│   └── params.yaml
├── .github/workflows/ci.yml
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_model_training.py
│   └── test_api.py
├── data/samples/*.csv                 # Jeux miniatures fictifs
├── docs/MLflow_RUNBOOK.md             # Démarrage tracking/registry
├── docs/PROJECT_ROADMAP.md            # Plan d’exécution guidé
└── requirements.txt
```

---

## 2. Installation & environnement

1. **Créer l’environnement** :
   ```bash
   python -m venv env
   .\env\Scripts\Activate.ps1  # PowerShell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Configurer MLflow** :
   ```bash
   mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
   ```
   - Adaptez les URI dans `configs/mlflow.yaml` ou via les variables d’environnement `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`.
3. **Préparer Git** :
   ```bash
   git init
   git add .
   git commit -m "chore: init credit scoring platform"
   git remote add origin git@github.com:<user>/<repo>.git
   git push -u origin main
   ```

---

## 3. Data science : étapes guidées

1. **Jointure multi-tables** (`Src/pipelines/join_datasets.py`)
   - Exécutez `python Src/pipelines/join_datasets.py` pour générer `data/joined_clients.csv` à partir des extraits `data/samples/*.csv` (ou des tables Home Credit situées dans `Src/Data/`).
   - Les agrégats clients/produits/statistiques sont sauvegardés automatiquement.

2. **Feature engineering & gestion du déséquilibre** (`Src/features/feature_engineering.py`)
   - Pipeline `ColumnTransformer` + `OneHotEncoder` + imputations.
   - Découpe stratifiée train/valid/test et export en Parquet sous `artifacts/features/`.
   - Calcul automatique des `sample_weight` pour les modèles sensibles au déséquilibre et sauvegarde du préprocesseur avec joblib.

3. **Score métier & GridSearchCV** (`Src/models/custom_score.py` + `Src/models/train_model.py`)
   - `business_cost_score` (FN coût 10× FP) et `optimal_threshold` déterminent le seuil métier.
   - `train_model.py` effectue GridSearchCV, logge les métriques dans MLflow, calcule SHAP, sauvegarde le modèle + seuil (`artifacts/models/`) puis publie la meilleure version dans le Model Registry.

4. **Explicabilité** :
   - `train_model.py` produit `artifacts/plots/shap_summary.png` loggé dans MLflow (`artifact_path=explainability`).
   - L’endpoint `/explain` de l’API renvoie les SHAP locaux via `shap.KernelExplainer`.

5. **Score personnalisé** :
   - Le seuil optimal calculé est écrit dans `artifacts/models/threshold.json` et consommé par l’API/Streamlit pour garder une cohérence métier.

---

## 4. MLOps & automatisation

- **MLflow tracking/registry/serving** : suivez `docs/MLflow_RUNBOOK.md` pour lancer le serveur, créer l’expérience `credit_scoring_prod`, valider la promotion Production et démarrer un endpoint de serving si besoin.
- **CI/CD GitHub Actions** (`.github/workflows/ci.yml`) : tests Pytest + lint `black`/`isort`, puis build des containers API/Streamlit. Le workflow s’exécute automatiquement sur `push`/`pull_request` vers `main`.
- **Tests Pytest** (`tests/`) : scénarios couvrant pipeline, entraînement (marqué `@pytest.mark.integration`) et API (`TestClient`). Exécutez `pytest --maxfail=1 --disable-warnings -q`.
- **Secrets & sécurité** : stockez les tokens MLflow/GitHub/Cloud dans les secrets GitHub Actions. Utilisez `python-dotenv` ou un coffre-fort (Azure Key Vault, AWS Secrets Manager) pour injecter les URI lors du déploiement.
- **Déploiement cloud automatisé** : build des images `docker build -f docker/Dockerfile.api -t credit-scoring-api .`, push vers votre registre (GHCR/ECR/ACR) puis déclenchez Render/Railway/Azure Container Apps via la CI.

---

## 5. API REST & Interface Streamlit

- **FastAPI** (`Api/app/main.py`)
  - Endpoints `GET /health`, `POST /predict`, `POST /explain`.
  - Chargement du modèle via `mlflow.pyfunc.load_model("models:/credit_scoring_model/Production")` et seuil `artifacts/models/threshold.json`.
  - Validation Pydantic (`ClientFeatures`), gestion d’erreur `HTTPException`, réponse JSON : probabilité, décision, seuil, explication SHAP.

- **Documentation API** :
  - Ouvrir `http://localhost:8000/docs` après `uvicorn Api.app.main:app --reload`.
  - Ajoutez un `schemas.py` si vous désirez séparer les modèles.

- **Streamlit** (`Interface/streamlit_app.py`)
  - Champ `client_id`, zone JSON du vecteur de features, bouton pour appeler `/predict` et case à cocher pour `/explain`.
  - Upload d’un rapport Evidently et bouton de téléchargement pour le partager.
  - Déploiement autonome via `streamlit run Interface/streamlit_app.py` ou via `docker/Dockerfile.streamlit`.

---

## 6. Monitoring & alertes

- **Rapports Evidently** (`Src/monitoring/drift_monitor.py`)
  - Charge les datasets de référence/production (`artifacts/features/X_valid.parquet` vs `X_test.parquet` par défaut).
  - Génère HTML + JSON sous `Monitoring/reports/`, calcul du `drift_share` et affichage d’une alerte (à rediriger vers Slack/Teams/Email).

- **Système d’alertes** :
  - Étendez `alert_if_needed` pour déclencher un webhook Slack.
  - Archivez les alertes dans `Monitoring/logs/alerts.log` (à créer).

- **Dashboard** :
  - Les rapports HTML peuvent être intégrés dans Streamlit (`st.components.v1.html`) ou importés dans PowerBI.

---

## 7. Tests & qualité

- `tests/test_data_pipeline.py` : vérifie les colonnes issues de `assemble_dataset`.
- `tests/test_api.py` : exemple `TestClient` FastAPI pour `/health`.
- `tests/test_model_training.py` : test d’intégration garantissant la production d’expériences MLflow et la sauvegarde du modèle.

Exécutez `pytest --maxfail=1 --disable-warnings -q`. Ajoutez `pytest-cov` et des tests de dérive si nécessaire.

---

## 8. Déploiement & exploitation

1. **Local** :
   ```bash
   uvicorn Api.app.main:app --reload --port 8000
   streamlit run Interface/streamlit_app.py
   ```
2. **Docker** : utilisez `docker/Dockerfile.api` & `docker/Dockerfile.streamlit`, build & run (docker-compose à créer si besoin).
3. **Cloud** : push des images vers un registre, déploiement sur Azure Container Apps / AWS ECS / Render, connexion MLflow via VPN ou service managé.

---

## 9. Exemples de données & reproductibilité

- Fichiers `data/samples/*` fournissent des données fictives pour tester la pipeline sans manipuler les tables Home Credit (déjà placées dans `Src/Data/`).
- Remplacez progressivement par vos propres données et ajustez les chemins.
- Documentez toute modification majeure dans `docs/EXECUTION_GUIDE.md` pour garder le fil rouge de la soutenance.

---

## 10. Bonnes pratiques & prochaines étapes

1. **Sécurité** : isolement des secrets, scanning `pip-audit`, HTTPS pour l’API, vérification des droits IAM.
2. **Performance** : profiling mémoire/CPU, caching des features, ajout d’un orchestrateur (Prefect/Airflow) pour scheduler la préparation & le monitoring.
3. **Extension MLOps** : feature store (Feast), triggers de re-training basés sur la dérive, dashboards Streamlit dédiés dans `Monitoring/dashboard/`.

---

## 11. Ordre d’exécution conseillé

1. `python Src/pipelines/join_datasets.py`
2. `python Src/features/feature_engineering.py`
3. `python Src/models/train_model.py` (après avoir lancé MLflow)
4. `uvicorn Api.app.main:app --reload`
5. `streamlit run Interface/streamlit_app.py`
6. `python Src/monitoring/drift_monitor.py`
7. `pytest --maxfail=1 --disable-warnings -q`
8. `git push origin main` pour déclencher la CI/CD Docker.

Utilisez `docs/PROJECT_ROADMAP.md` comme checklist et suivez ce sequence pour garantir la reproductibilité.
