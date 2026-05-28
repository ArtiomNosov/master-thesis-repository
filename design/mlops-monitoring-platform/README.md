# MLOps Monitoring Platform UI Mockup

Документация к high-fidelity макету MLOps-платформы для мониторинга качества ML-моделей, data drift, model drift, алертов и состояния моделей в production.

## Что находится в директории

- `index.html` — интерактивный HTML-board с шестью desktop-экранами 1440px.
- `styles.css` — дизайн-токены и reusable UI-компоненты.
- `exports/png/` — PNG-картинки всех экранов 1440x1024.
- `figma-plugin/` — локальный Figma plugin generator для создания редактируемых фреймов внутри Figma.
- `scripts/export-png.sh` — скрипт экспорта HTML-экранов в PNG через headless Chrome (Linux/macOS).
- `scripts/export-png.ps1` — тот же экспорт для Windows/PowerShell.

## Экраны макета

1. `01-main-mlops-dashboard.png` — главный MLOps-дашборд.
2. `02-model-detail.png` — карточка модели.
3. `03-data-drift-monitoring.png` — мониторинг дрейфа данных.
4. `04-model-performance.png` — производительность модели.
5. `05-alerts-incidents.png` — оповещения и инциденты.
6. `06-model-registry.png` — реестр моделей.

## Состав интерфейса

Макет выполнен в стиле clean enterprise SaaS, light mode:

- левая навигация: Дашборд, Модели, Дрейф данных, Производительность, Оповещения, Реестр, Настройки;
- верхняя панель: поиск, период, окружение, уведомления, профиль;
- карточки метрик и системного статуса;
- таблицы активных моделей, features, incidents и registry versions;
- графики качества, drift distributions, latency/throughput и business impact;
- status badges: Норма, Предупреждение, Критично, Инфо, Архив;
- alert cards с severity, причиной, затронутой моделью, временем, статусом и быстрыми действиями;
- realistic MLOps data для `churn_prediction_v3`, `fraud_detection_v2`, `demand_forecast_v1`, `credit_risk_v4`.

## Как открыть HTML-макет

Откройте файл `index.html` в браузере:

```bash
xdg-open design/mlops-monitoring-platform/index.html
```

Можно открыть отдельный экран в export-режиме:

```text
index.html?screen=dashboard
index.html?screen=model-detail
index.html?screen=data-drift
index.html?screen=model-performance
index.html?screen=alerts-incidents
index.html?screen=model-registry
```

## Как пересобрать PNG-картинки

```powershell
design/mlops-monitoring-platform/scripts/export-png.ps1
```

```bash
design/mlops-monitoring-platform/scripts/export-png.sh
```

Скрипт сохраняет PNG в:

```text
design/mlops-monitoring-platform/exports/png/
```

Каждый файл экспортируется в размере 1440x1024.

## Figma API: что реально можно сделать

Проверенные API Figma:

- REST API: `https://api.figma.com`
- REST API документация: `https://developers.figma.com/docs/rest-api/`
- Plugin API `figma.createFrame()`: `https://developers.figma.com/docs/plugins/api/properties/figma-createframe/`

Важное ограничение: Figma REST API позволяет читать существующие файлы, получать JSON-дерево слоёв, экспортировать изображения, работать с комментариями, компонентами, стилями, проектами и webhooks. Он не предоставляет endpoint для создания нового дизайн-файла или программного добавления фреймов на canvas.

Для создания редактируемых фреймов используется Figma Plugin API внутри авторизованной Figma-сессии. Поэтому в репозитории добавлен локальный plugin generator, который создаёт макет через методы Plugin API, включая `figma.createFrame()`, `figma.createRectangle()` и `figma.createText()`.

## Как создать редактируемый Figma-файл

1. Откройте Figma в браузере или desktop app.
2. Создайте новый пустой файл или откройте существующий проект.
3. Перейдите в `Plugins -> Development -> Import plugin from manifest...`.
4. Выберите файл:

```text
design/mlops-monitoring-platform/figma-plugin/manifest.json
```

5. Запустите plugin `ModelPulse MLOps Dashboard Generator`.

После запуска plugin создаст шесть редактируемых 1440px Figma-фреймов с компонентами, таблицами, графиками, бейджами, карточками алертов, sidebar navigation и topbar controls.

## Почему здесь нет live Figma-ссылки

Cursor Web/Cloud Agent не имеет авторизованной Figma-сессии, Figma MCP и `FIGMA_TOKEN`. Без такого доступа нельзя создать live Figma-проект и получить публичную редактируемую ссылку из среды агента.

Вместо этого подготовлены:

- готовый HTML high-fidelity макет;
- PNG-экспорты всех экранов;
- Figma plugin generator для создания редактируемого файла уже внутри Figma.
