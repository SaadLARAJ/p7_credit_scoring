# P8 - Dashboard Interactif & Veille Technologique

Ce dossier contient les livrables du projet P8 pour **Prêt à Dépenser**, suite logique du P7 (Credit Scoring).

## 📋 Contenu du projet

### 1. Dashboard Interactif (`P8_Dashboard/`)

Application Streamlit multipage permettant aux chargés de relation client d'expliquer les décisions d'octroi de crédit de manière transparente.

**Pages disponibles :**
| Page | Description |
|------|-------------|
| 🎯 Score Client | Visualisation du score de crédit avec jauge accessible |
| 🔍 Explicabilité | Analyse SHAP (importance locale et globale) |
| 📊 Comparaison | Histogrammes client vs population |
| 📈 Analyse Bivariée | Scatter plots entre deux variables |
| ✏️ Simulation | Modification des features et re-scoring (optionnel) |

**Accessibilité WCAG :**
- ✅ 1.1.1 Contenu non textuel (descriptions textuelles)
- ✅ 1.4.1 Utilisation de la couleur (texte + icône + couleur)
- ✅ 1.4.3 Contraste minimum (4.5:1)
- ✅ 1.4.4 Redimensionnement du texte (200%)
- ✅ 2.4.2 Titre de page (unique par page)

### 2. Veille Technologique (`P8_Research/`)

Étude et implémentation de TabNet comme alternative au LightGBM du P7.

**Contenu :**
- `notebooks/` : Notebooks de recherche et POC
- `docs/note_methodologique.md` : Note méthodologique complète
- `src/` : Code source pour TabNet et comparaison

---

## 🚀 Démarrage rapide

### Dashboard

```bash
# Installation des dépendances
cd P8_Dashboard
pip install -r requirements.txt

# Lancement en local
streamlit run app.py
```

Le dashboard appelle l'API P7 déployée sur Render :
- API : https://p7-credit-scoring-2.onrender.com/docs

### Veille technologique

```bash
# Installation des dépendances
cd P8_Research
pip install -r requirements.txt

# Ouvrir les notebooks
jupyter notebook
```

---

## 📁 Structure du projet

```
P8_Dashboard/
├── app.py                      # Point d'entrée multipage
├── pages/
│   ├── 1_🎯_Score_Client.py    # Score et décision
│   ├── 2_🔍_Explicabilite.py   # SHAP waterfall
│   ├── 3_📊_Comparaison.py     # Histogrammes
│   ├── 4_📈_Analyse_Bivariee.py# Scatter plots
│   └── 5_✏️_Simulation.py      # Modification client
├── components/
│   ├── gauge.py                # Jauge accessible
│   ├── shap_charts.py          # Visualisations SHAP
│   ├── comparison_charts.py    # Histogrammes et scatter
│   └── accessibility.py        # Helpers WCAG
├── utils/
│   ├── api_client.py           # Client API P7
│   └── data_loader.py          # Chargement données
├── .streamlit/
│   └── config.toml             # Thème accessible
└── requirements.txt

P8_Research/
├── notebooks/
│   ├── 01_research_bibliography.ipynb  # Sources et état de l'art
│   ├── 02_tabnet_poc.ipynb             # Implémentation POC
│   └── 03_comparison_analysis.ipynb    # Comparaison
├── docs/
│   └── note_methodologique.md   # Note méthodologique
├── src/
│   ├── tabnet_model.py          # Wrapper TabNet
│   └── comparison_utils.py      # Utilitaires comparaison
└── requirements.txt
```

---

## 🔗 Liens utiles

- **Dashboard P8** : *À déployer sur Streamlit Cloud*
- **API P7** : https://p7-credit-scoring-2.onrender.com/docs
- **GitHub** : https://github.com/SaadLARAJ/p7_credit_scoring

---

## 📊 Métriques et évaluation

### Dashboard
- Parcours utilisateur clair (CE1)
- Graphiques interactifs (CE2)
- Graphiques lisibles (CE3)
- Réponse à la problématique métier (CE4)
- Accessibilité WCAG (CE5)
- Déploiement cloud (CE6)

### Veille technologique
- Sources reconnues (CE1)
- Détails mathématiques (CE2)
- Preuve de concept avec comparaison (CE3)

---

## 👤 Auteur

**Saad LARAJ** - Projet OpenClassrooms Data Scientist

---

## 📝 Licence

Ce projet est à usage éducatif dans le cadre de la formation OpenClassrooms.
