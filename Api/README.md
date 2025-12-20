# Guide API & Streamlit

Ce dossier comportera deux livrables :
1. `app/main.py` – service FastAPI exposant le modèle de scoring
2. `streamlit_app.py` – interface analyste consommant l’API

## Structure attendue
```
Api/
├── app/
│   ├── __init__.py
│   ├── main.py             # Point d’entrée FastAPI
│   ├── schemas.py          # Modèles Pydantic requête/réponse
│   ├── dependencies.py     # Chargement du modèle, seuil métier, logger
│   └── config.py           # URI MLflow, nom du modèle, chemins d’artefacts
├── streamlit_app.py        # Client interactif
├── requirements.txt        # Optionnel (hérite du requirements principal)
└── README.md               # (ce fichier)
```

## Attendus FastAPI
- **Chargement du modèle** – `mlflow.pyfunc.load_model("models:/pret_credit_scoring/Production")` au démarrage. Stocker le seuil métier dans l’artefact du modèle (JSON) afin que l’API applique la même coupure que Notebook 02.
- **Endpoints**
  - `GET /health` → `{ "status": "ok", "model_version": "Production vX" }`
  - `POST /predict` → une requête contenant un client (dict) ou un lot (liste). La réponse retourne probabilité, décision, seuil, explications locales (top features SHAP) et impact coût métier.
- **Gestion des erreurs** – valider les champs obligatoires et renvoyer `422` si le schéma n’est pas respecté.
- **Tests unitaires** – `pytest` + `fastapi.testclient.TestClient` (voir `tests/test_api.py`).

## Streamlit
- Formulaire pour les principales features ou téléchargement d’un JSON généré par Notebook 02.
- Bouton envoyant la requête via `requests.post()`.
- Affichage probabilité, décision, coût, visuel SHAP optionnel.

## Déploiement
1. **Image container** – créer un `Dockerfile` à la racine installant les requirements et copiant le code API.
2. **Workflow GitHub Actions `deploy_api.yml`**
   - Déclenché sur `main` ou tags.
   - Étapes : checkout → setup Python → installer deps → pytest → build + push container → appel API Render/Railway/Azure pour déployer.
3. **Secrets** – stocker URI MLflow, nom du modèle, clés cloud dans les secrets GitHub.
4. **Smoke test local** – `python -m uvicorn app.main:app --reload` puis `Invoke-RestMethod` ou `curl`.

## Exemple de payload
```json
{
  "client": {
    "SK_ID_CURR": 100001,
    "EXT_SOURCE_2": 0.65,
    "AMT_CREDIT": 406597.5,
    "DAYS_BIRTH": -16704,
    "DAYS_EMPLOYED": -13870,
    "BUREAU_ACTIVE_LOAN_COUNT_MEAN": 1.2,
    "PAYMENT_RATE": 0.055,
    "...": "..."
  }
}
```
Adapter la liste en fonction des features produites par le Notebook 01.

## Documentation à livrer
- Mettre à jour ce README lorsque les endpoints seront actifs (URL, exemples `curl`).
- Ajouter un diagramme d’architecture (Streamlit → API → registry MLflow → base de données).
- Lister des conseils de dépannage (renouvellement de token, authentification MLflow, pinning des dépendances).
