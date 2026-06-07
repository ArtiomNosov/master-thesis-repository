# Figma API: краткое исследование для макета MLOps Dashboard

## Цель

Проверить, можно ли из Cursor Web/Cloud Agent создать настоящий редактируемый Figma-файл и получить live Figma-ссылку программно.

## Найденные API

### 1. Figma REST API

- Base URL: `https://api.figma.com`
- Документация: `https://developers.figma.com/docs/rest-api/`
- Аутентификация: personal access token или OAuth2.

REST API подходит для:

- чтения существующего Figma-файла: `GET /v1/files/:key`;
- получения отдельных nodes: `GET /v1/files/:key/nodes`;
- экспорта изображений из существующего файла: `GET /v1/images/:key`;
- работы с комментариями;
- получения проектов, команд, компонентов, стилей, переменных и webhooks.

Ограничение: REST API не создаёт новый дизайн-файл и не добавляет фреймы/слои на canvas.

### 2. Figma Plugin API

- Документация `createFrame`: `https://developers.figma.com/docs/plugins/api/properties/figma-createframe/`

Plugin API подходит для создания и редактирования canvas внутри открытого Figma-файла:

```js
const frame = figma.createFrame();
frame.x = 50;
frame.y = 50;
frame.resize(1440, 1024);
```

Именно этот путь использован в `figma-plugin/code.js`: plugin создаёт шесть desktop-фреймов, текстовые слои, прямоугольники, карточки, таблицы, бейджи, графики и кнопки.

## Вывод

Из Cursor Web без авторизованной Figma-сессии, Figma MCP или Figma token нельзя напрямую создать live Figma project link.

Рабочий путь:

1. Сгенерировать макет и plugin в репозитории.
2. Открыть Figma пользователем с доступом к нужному workspace.
3. Импортировать `figma-plugin/manifest.json`.
4. Запустить plugin.
5. Получить live Figma-ссылку уже из Figma через Share.

## Что уже подготовлено

- `index.html` — high-fidelity HTML-board.
- `exports/png/` — картинки всех экранов 1440x1024.
- `figma-plugin/manifest.json` и `figma-plugin/code.js` — generator для редактируемого Figma-файла.
