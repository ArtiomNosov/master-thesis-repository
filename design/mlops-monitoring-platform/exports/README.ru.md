# PNG-экспорты экранов платформы мониторинга моделей

В директории `png/` сохранены изображения всех требуемых экранов в размере 1440×1024. Видимый текст на всех экранах — на русском языке без латиницы.

1. `01-main-mlops-dashboard.png` — главная панель мониторинга моделей.
2. `02-model-detail.png` — страница конкретной модели.
3. `03-data-drift-monitoring.png` — экран мониторинга дрейфа данных.
4. `04-model-performance.png` — экран производительности модели.
5. `05-alerts-incidents.png` — экран оповещений и инцидентов.
6. `06-model-registry.png` — экран реестра моделей.

Перед пересборкой проверьте отсутствие латиницы в HTML-макете:

```powershell
design/mlops-monitoring-platform/scripts/check-no-latin.ps1
```

Чтобы пересобрать изображения из HTML-макета, выполните:

```powershell
design/mlops-monitoring-platform/scripts/export-png.ps1
```

```bash
design/mlops-monitoring-platform/scripts/export-png.sh
```
