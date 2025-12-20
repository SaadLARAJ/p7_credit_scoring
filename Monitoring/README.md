## Monitoring & Alerting

Ce dossier reçoit les rapports Evidently exportés par `Src/monitoring/drift_monitor.py` ainsi que les captures transmises à l'équipe risque. Les sous-dossiers proposés dans le README principal sont :

- `Monitoring/reports/` : rapports HTML/JSON pour la dérive.
- `Monitoring/logs/` : alertes textuelles postées par les scripts automatiques.
- `Monitoring/dashboard/` : fichiers Streamlit/PowerBI pour visualiser l'historique de drift.

Les rapports contenus ici peuvent être versionnés (petits fichiers) ou envoyés vers un bucket objet lorsque la volumétrie augmente. Pensez à nettoyer régulièrement pour éviter les données sensibles dans Git.
