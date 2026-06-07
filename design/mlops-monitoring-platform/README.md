# Макет платформы мониторинга моделей машинного обучения

Документация к high-fidelity макету платформы мониторинга моделей машинного обучения: качество моделей, дрейф данных, дрейф моделей, оповещения и состояние моделей в промышленной среде.

## Что находится в директории

- `index.html` — интерактивный HTML-board с шестью desktop-экранами 1440px.
- `styles.css` — дизайн-токены и переиспользуемые UI-компоненты.
- `exports/png/` — PNG-картинки всех экранов 1440×1024.
- `figma-plugin/` — локальный генератор Figma plugin для создания редактируемых фреймов внутри Figma.
- `scripts/export-png.sh` — скрипт экспорта HTML-экранов в PNG через headless Chrome (Linux/macOS).
- `scripts/export-png.ps1` — тот же экспорт для Windows/PowerShell.
- `scripts/check-no-latin.ps1` — проверка отсутствия латиницы в видимом тексте HTML-макета.

## Экраны макета

1. `01-main-mlops-dashboard.png` — главная панель мониторинга моделей.
2. `02-model-detail.png` — карточка модели.
3. `03-data-drift-monitoring.png` — мониторинг дрейфа данных.
4. `04-model-performance.png` — производительность модели.
5. `05-alerts-incidents.png` — оповещения и инциденты.
6. `06-model-registry.png` — реестр моделей.

## Состав интерфейса

Макет выполнен в стиле clean enterprise SaaS, light mode. Весь видимый текст — на русском языке без латиницы и англицизмов.

- левая навигация: Панель мониторинга, Модели, Дрейф данных, Производительность, Оповещения, Реестр, Настройки;
- верхняя панель: поиск, период, окружение, уведомления, профиль;
- карточки метрик и системного статуса;
- таблицы активных моделей, признаков, инцидентов и версий реестра;
- графики качества, распределений дрейфа, задержки/пропускной способности и бизнес-эффекта;
- статусные бейджи: Норма, Предупреждение, Критично, Инфо, Архив;
- карточки оповещений с уровнем критичности, причиной, затронутой моделью, временем, статусом и быстрыми действиями;
- реалистичные данные для `прогноз_оттока_3`, `обнаружение_мошенничества_2`, `прогноз_спроса_1`, `кредитный_риск_4`.

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

Перед экспортом рекомендуется проверить отсутствие латиницы:

```powershell
design/mlops-monitoring-platform/scripts/check-no-latin.ps1
```

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

Каждый файл экспортируется в размере 1440×1024.

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

5. Запустите plugin «Генератор панели мониторинга ПульсМоделей».

После запуска plugin создаст шесть редактируемых 1440px Figma-фреймов с компонентами, таблицами, графиками, бейджами, карточками оповещений, боковой навигацией и элементами верхней панели.

## Почему здесь нет live Figma-ссылки

Cursor Web/Cloud Agent не имеет авторизованной Figma-сессии, Figma MCP и `FIGMA_TOKEN`. Без такого доступа нельзя создать live Figma-проект и получить публичную редактируемую ссылку из среды агента.

Вместо этого подготовлены:

- готовый HTML high-fidelity макет;
- PNG-экспорты всех экранов;
- Figma plugin generator для создания редактируемого файла уже внутри Figma.
