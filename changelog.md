# Changelog

## 2026-05-25

- Добавлен расчет экономического обоснования внедрения в текст ВКР: исходные метрики времени и качества, денежные показатели, срок окупаемости и ROI.
- Обновлена презентация ВКР: добавлены слайды с базовыми метриками сравнения человека и модели, а также с денежными метриками внедрения.
- Добавлен high-fidelity HTML-макет MLOps-платформы мониторинга (light mode) в `design/mlops-monitoring-platform/`.
- Спроектированы шесть desktop-экранов 1440px: Dashboard, Model Detail, Data Drift Monitoring, Model Performance, Alerts and Incidents, Model Registry.
- Добавлены переиспользуемые UI-компоненты и стили: sidebar navigation, topbar controls, metric cards, status badges, alert cards, tables, charts, recommendations, registry actions.
- Макет заполнен реалистичными MLOps-данными для `churn_prediction_v3`, `fraud_detection_v2`, `demand_forecast_v1` и связанных сценариев качества моделей, drift, инцидентов и registry.
- Добавлен локальный Figma plugin generator, создающий редактируемые 1440px-фреймы MLOps dashboard внутри Figma при запуске в авторизованной Figma desktop/web-сессии.

## 2026-05-26

- Документация артефакта MLOps dashboard переработана на русском языке.
- Добавлены русскоязычные заметки по Figma API с пояснением ограничений REST API и пути через Plugin API для создания редактируемых Figma-фреймов.
- Добавлены PNG-экспорты всех шести экранов 1440x1024 в `design/mlops-monitoring-platform/exports/png/`.
- Добавлен скрипт экспорта PNG из HTML-макета через headless Chrome.

## 2026-05-27

- Интерфейс HTML-макета, Figma plugin generator и PNG-экспорты переведены на русский язык.
- Пересобраны все шесть PNG 1440x1024 из обновлённого HTML-макета.
- Добавлен Windows-скрипт `scripts/export-png.ps1` для локальной пересборки PNG.
