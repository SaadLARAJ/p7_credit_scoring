# Feuille de route – Plateforme de credit scoring "Prêt à Dépenser"
Checklist détaillée expliquant **quoi faire** et **pourquoi** pour chaque livrable.

## Phase 0 – Fondations
1. **Créer/activer l’environnement virtuel (`env/`)** – isole les dépendances (MLflow, LightGBM) et évite les conflits sur la machine.
2. **Installer `requirements.txt`** – garantit que notebooks, API et monitoring utilisent exactement les mêmes versions.
3. **Initialiser Git + `.gitignore`** – indispensable pour travailler en équipe, déclencher la CI/CD et tracer l’historique exigé par Mickael.
4. **Télécharger les données Kaggle** – déjà présentes sous `Src/Data/`; considérez-les comme *read-only* pour assurer la reproductibilité.

## Phase 1 – Adaptation du kernel Kaggle
1. **Lire le kernel de João Saguiar (LightGBM with simple features)** – noter toutes les agrégations et astuces avant de les réimplémenter localement.
2. **Notebook `01_data_preparation.ipynb`**
   - Reproduire les blocs de features (bureau, bureau_balance, previous_application, POS, installments, credit card) en surveillant la mémoire (<16 Go).
   - Ajouter des cellules Markdown expliquant chaque adaptation (filtres différents, typages, optimisation).
   - Sauvegarder les datasets traités dans `Src/Prepared/` (train/valid/test en parquet) pour accélérer les itérations suivantes.
   - Pourquoi : cette étape garantit un pipeline reproductible pour MLflow et la production.
3. **Contrôles exploratoires** – distribution de la cible, gestion du déséquilibre, corrélations pour éviter toute fuite de données.

## Phase 2 – Modélisation avec métrique métier + MLflow
1. **Démarrer MLflow (voir `docs/MLflow_RUNBOOK.md`)** avant d’exécuter le Notebook 02.
2. **Notebook `02_model_training_mlflow.ipynb`**
   - Implémenter la fonction de coût : `coût = 10*FN + 1*FP`.
   - Optimiser le seuil de décision (balayage 0→1 ou fonction personnalisée `make_scorer`).
   - Baselines : régression logistique + LightGBM par défaut pour valider la pipeline.
   - Recherche d’hyperparamètres : `GridSearchCV` ou `Optuna`, avec journalisation MLflow de chaque essai.
   - Enregistrer le meilleur modèle dans le Model Registry (`pret_credit_scoring`).
3. **Explicabilité**
   - Importances globales (gain LightGBM, permutation) + SHAP global.
   - Explications locales (SHAP/LIME) pour quelques clients, loguées comme artefacts MLflow.

## Phase 3 – Serving & Streamlit
1. **Implémenter FastAPI (`Api/app/main.py`)**
   - Charger le modèle Production depuis MLflow (`mlflow.pyfunc.load_model`).
   - Endpoints `/health` et `/predict` retournant probabilité + décision + seuil utilisé.
   - Pourquoi : l’API constitue le contrat avec les applications aval.
2. **Tests (`tests/test_api.py`)** – utiliser FastAPI TestClient pour vérifier schémas et logique métier.
3. **Client Streamlit**
   - Formulaire pour saisir les principales features ou coller un JSON.
   - Appel de l’API, affichage probabilité/décision/coût, option SHAP.

## Phase 4 – Automatisation MLOps
1. **Dépôt GitHub** – pousser le code, activer la protection de branche, imposer revue/CI.
2. **GitHub Actions**
   - `ci.yml` : lint (black/isort) + pytest (unitaires, API) à chaque push.
   - `deploy_api.yml` : sur tag, construire l’image (Docker) et déclencher le déploiement Render/Railway/Azure.
3. **MLflow Model Registry & Serving** – scripts pour promouvoir les modèles et lancer `mlflow models serve` lors des vérifications.
4. **Preuves** – capture d’écran MLflow UI à intégrer dans le support de soutenance.

## Phase 5 – Monitoring & Data Drift
1. **Notebook `03_monitoring_drift.ipynb`** – comparer `application_train` (référence) et `application_test` (production) avec Evidently ; sauvegarder `mlops/data_drift_report.html`.
2. **Analyse** – commenter les features en drift, proposer une stratégie d’alerte (seuils, fréquence d’exécution, actions correctives).

## Phase 6 – Préparation finale
1. **Documentation** – compléter le README avec les URL finales (API cloud, Streamlit, screenshot MLflow, lien drift report).
2. **Support de soutenance** – intégrer pipeline, métriques clés, importances de features, plan de monitoring et limites.
3. **Répétition** – préparer l’argumentaire (choix des modèles, gestion du déséquilibre/cost-sensitive, monitoring, questions pour Mickael).
