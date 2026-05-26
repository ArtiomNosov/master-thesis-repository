# PNG-экспорты экранов MLOps Monitoring Platform

В директории `png/` сохранены изображения всех требуемых экранов в размере 1440x1024:

1. `01-main-mlops-dashboard.png` — главный MLOps Dashboard.
2. `02-model-detail.png` — страница конкретной модели.
3. `03-data-drift-monitoring.png` — экран Data Drift Monitoring.
4. `04-model-performance.png` — экран Model Performance.
5. `05-alerts-incidents.png` — экран Alerts and Incidents.
6. `06-model-registry.png` — экран Model Registry.

Чтобы пересобрать изображения из HTML-макета, выполните:

```bash
design/mlops-monitoring-platform/scripts/export-png.sh
```
