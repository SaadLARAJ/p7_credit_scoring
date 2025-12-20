# Guide d’exécution – Commandes & Checklist
Rejouer le projet de bout en bout en suivant ces étapes.

## 1. Préparer l’environnement
```powershell
python -m venv env
./env/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Préparer Git
```powershell
git init
copy .\.gitignore.template .\.gitignore  # si besoin, sinon utiliser le .gitignore fourni
git add .
git commit -m "chore: bootstrap credit scoring"
```
Créer le dépôt GitHub → `git remote add origin ...` → `git push -u origin main`.

## 3. Lancer le serveur MLflow
```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_REGISTRY_URI = $env:MLFLOW_TRACKING_URI
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns_artifacts --host 127.0.0.1 --port 5000
```
Garder ce terminal ouvert ; exécuter les notebooks dans un autre.

## 4. Ordre d’exécution des notebooks
1. `jupyter lab` ou `jupyter notebook`
2. `Notebook/01_data_preparation.ipynb` → génère les fichiers parquet.
3. `Notebook/02_model_training_mlflow.ipynb` (avec MLflow en route) → enregistre le modèle.
4. `Notebook/03_monitoring_drift.ipynb` → produit `mlops/data_drift_report.html`.

## 5. Tests locaux de l’API
```powershell
uvicorn Api.app.main:app --reload
```
Une fois `/predict` codé, tester via PowerShell ou Streamlit :
```powershell
streamlit run Api/streamlit_app.py
```

## 6. Pytest & lint
```powershell
black .
isort .
pytest --maxfail=1 --disable-warnings -q
```

## 7. GitHub Actions
- Push sur `main` : déclenche CI (lint + tests).
- Tag `v1.0.0` : déclenche CD (build image + déploiement cloud de l’API).

## 8. Smoke test MLflow Serving
```powershell
mlflow models serve -m "models:/pret_credit_scoring/Production" -p 1234 --no-conda
```
Envoyer un échantillon pour vérifier que l’artefact se charge avant la mise en prod.

## 9. Livrables pour la soutenance
- Capture MLflow UI (runs, métriques, artefacts).
- Rapport Evidently `mlops/data_drift_report.html`.
- URL de l’API déployée + lien Streamlit public.
- Slides résumant processus, décisions, métriques, importances de features, monitoring.

Mettez ce guide à jour si vous changez d’hébergement ou ajoutez de l’automatisation.
