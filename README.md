# master-thesis-repository

## I. Important Links

### 1. [Google Drive](https://drive.google.com/drive/folders/14wGy7ov1FCh4fVoLwjFcMb3xYfXxYsLm?usp=drive_link)

### 2. [Figma](https://www.figma.com/design/43z63QWcWMAaGqowHk1Yp1/master-thesis-design?node-id=11-1833&t=uW62QYp3NzVyESvQ-1)

## II. Reqcore ATS UI

В репозитории добавлен минимальный статический интерфейс ATS / Reqcore на русском языке.
Он не меняет ML-логику, API-контракты и Python-прототипы ранжирования.

### Запуск

```bash
cd ats-ui
npm run build
npm run start
```

Локальный URL интерфейса: <http://127.0.0.1:4173>.

### Проверки

```bash
cd ats-ui
npm run lint
npm run build
npm run test
npm run screenshots
```

Скриншоты и DOM-отчёт проверки русской локализации сохраняются в
`artifacts/i18n-ru-screenshots/`.
