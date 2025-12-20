# Runbook MLflow
Commandes pratiques pour lancer le serveur de tracking, ouvrir l’UI, journaliser les expériences, enregistrer les modèles et tester le serving.

## 1. Variables d’environnement
À définir par session (exemple PowerShell) :
```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_REGISTRY_URI = $env:MLFLOW_TRACKING_URI
```
Possibilité de les stocker dans un fichier `.env` et de les charger avec `python-dotenv`.

## 2. Démarrer le serveur MLflow + UI
```powershell
mlflow server `
    --backend-store-uri sqlite:///mlruns.db `
    --default-artifact-root ./mlruns_artifacts `
    --host 127.0.0.1 --port 5000
```
- `mlruns.db` conserve paramètres/metrics (SQLite).
- `mlruns_artifacts/` contient tous les artefacts (plots, modèles, JSON).
- L’UI est disponible sur `http://127.0.0.1:5000` (à capturer pour la soutenance).

## 3. Journalisation dans les notebooks
Dans le Notebook 02, enveloppez vos entraînements :
```python
with mlflow.start_run(run_name="lightgbm_grid"):
    mlflow.log_params({...})
    mlflow.log_metrics({"auc": auc, "business_cost": cost, "threshold": best_threshold})
    mlflow.log_artifact("artifacts/confusion_matrix.png")
    mlflow.lightgbm.log_model(
        lgbm_model,
        artifact_path="model",
        registered_model_name="pret_credit_scoring",
    )
```
Important : loguer le seuil métier optimisé et le sauvegarder avec le modèle (pickle ou JSON) afin que l’API applique exactement le même cut-off.

## 4. Workflow du Model Registry
1. Après un run, ouvrez l’UI → **Artifacts → Register Model** (`pret_credit_scoring`).
2. Promotion des stages :
   - `None` → `Staging` après validation.
   - `Staging` → `Production` une fois l’API validée.
3. Mentionnez les IDs de run dans vos commits Git pour garder la traçabilité.

## 5. Test de serving local
```powershell
mlflow models serve -m "models:/pret_credit_scoring/Production" -p 1234 --no-conda
```
Puis tester :
```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:1234/invocations `
  -ContentType 'application/json' `
  -Body '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```
Vous devez obtenir probabilité + classe ; cette étape précède la containerisation FastAPI.

## 6. Astuces de tracking
- `mlflow.set_experiment("credit_scoring")` pour séparer ce projet.
- Logger les métriques métiers par fold pour comparaisons fines.
- Ajouter les visualisations SHAP comme artefacts.
- Définir des tags (`source=joao_saguiar_kernel_adapted`, `data_version=2025-11-16`) pour assurer la gouvernance.

## 7. Sauvegarde
- Stopper le serveur avec `Ctrl+C`.
- Sauvegarder `mlruns.db` et `mlruns_artifacts/` avant tout changement majeur (migration, nettoyage…).
