# Changelog

## 2026-05-29

- E2E i18n: `npm run db:reseed`, dev на **http://localhost:3001** (3000 занят Langfuse), `dom-language-report.json` → `pagesWithUnexpectedLatin: []`, 24 скриншота обновлены.
- Исправлена загрузка локали `ru`: экранирован `@` в `emailPlaceholder` (`colleague{'@'}company.com`), иначе vue-i18n не компилировал `ru.json` и UI показывал ключи i18n.
- Расширена русификация Reqcore: `ru.json` (settings, pipeline, onboarding, auth), `useLocalizedEnums`, `$t()` в SettingsSidebar и dashboard pipeline.
- Русский демо-seed: вакансии, кандидаты (ФИО), вопросы анкеты, заметки по откликам; org «Демо Reqcore».
- `RUNBOOK-RU.md`, доработан `russian-ui.client.ts` и `scripts/capture-i18n-screenshots.mjs` (localhost, UI-login fallback, fail при видимых i18n-ключах).
- Проверки: `npm run build`, `vitest` (361 тест), `db:reseed` — успешно.

## 2026-05-27

- Подключён git submodule `reqcore` → https://github.com/ArtiomNosov/reqcore, ветка `thesis/ru-localization`.
- В форке Reqcore: русская локаль `i18n/locales/ru.json`, язык по умолчанию `ru`, плагин `app/plugins/russian-ui.client.ts`, навигация через `$t('nav.*')`, убраны пользовательские формулировки с «ATS».
- Docker: Postgres + MinIO подняты, выполнены `db:migrate` и `db:seed`, проверен вход `demo@reqcore.com` / `demo1234` и активация org `reqcore-demo`.
- Скриншоты dashboard (22 страницы): `reqcore/artifacts/i18n-ru-screenshots/`, отчёт `dom-language-report.json`.
- Исправлен `russian-ui.client.ts` (больше не ломает гидратацию dashboard).
- Нейтрализованы упоминания SuperJob в текстах ВКР и презентации (формулировка «агрегатор вакансий»).
- Термины приведены к виду «независимый энкодер (bi-encoder)»; в таблице задачи 1 убран дублирующий столбец «Комментарий».
- Вспомогательный скрипт агента перенесён в `.agents/artifacts/patch_vkr_texts.py` (не часть ВКР).
- В сравнительном подразделе заменены англицизмы «эмбеддинг» и сокращения LTR на русские формулировки с полными английскими названиями в скобках (markdown и DOCX).
- Уточнена терминология сравнения ранжирования: полные русские названия вместо сокращений BERT/Sentence-BERT/bi-encoder.
- Переформулированы фрагменты про согласованность вакансии и резюме и обоснование выбора трансформерного подхода на первом этапе работы.
- Слияние с `semester-4`: сохранён сравнительный подраздел в DOCX, подтянуты правки baseline и нейтрализации источника данных.
- Интерфейс HTML-макета, Figma plugin generator и PNG-экспорты переведены на русский язык.
- Пересобраны все шесть PNG 1440x1024 из обновлённого HTML-макета.
- Добавлен Windows-скрипт `scripts/export-png.ps1` для локальной пересборки PNG.
- Полная русификация макета мониторинга моделей: убрана латиница и англицизмы из видимого UI (HTML, Figma plugin, PNG).
- Добавлен скрипт `scripts/check-no-latin.ps1` для автоматической проверки отсутствия латиницы в HTML-макете.

## 2026-05-26

- Исправлены сравнения метрик в основном документе ВКР `vkr/artifacts/М24-534_НосовАИ_ВКР.docx`: контрольным ориентиром явно названа базовая модель `cointegrated/rubert-tiny2` без fine-tuning, добавлено объяснение объективности такого baseline и ограничения офлайн-оценки.
- Обновлена презентация `vkr/artifacts/М24-534_НосовАИ_презентация_ВКР.pptx`: удалены формулировки `+281%` и «ниже качества дообученной модели на той же val-выборке», добавлены фактические baseline-значения F1 и accuracy, а выводы смягчены до офлайн-сравнения с контрольной моделью.
- Согласованы Markdown-источники презентации и ВКР: `docs/obsidian/presentation_defense.md`, `docs/obsidian/presentation_plan.md`, `docs/obsidian/thesis/[USED] thesis_draft_final_ru.md`, `docs/obsidian/Baseline_Comparison_Report.md`.
- Исправленный сравнительный фрагмент перенесен в основной файл ВКР `vkr/artifacts/М24-534_НосовАИ_ВКР.docx` перед разделом 4.1.
- В DOCX обновлено количество источников и добавлены источники по векторной модели поиска, TF-IDF/term specificity, PageRank и LambdaMART.
- Документация артефакта MLOps dashboard переработана на русском языке.
- Добавлены русскоязычные заметки по Figma API с пояснением ограничений REST API и пути через Plugin API для создания редактируемых Figma-фреймов.
- Добавлены PNG-экспорты всех шести экранов 1440x1024 в `design/mlops-monitoring-platform/exports/png/`.
- Добавлен скрипт экспорта PNG из HTML-макета через headless Chrome.

## 2026-05-25

- Добавлен high-fidelity HTML-макет MLOps-платформы мониторинга (light mode) в `design/mlops-monitoring-platform/`.
- Спроектированы шесть desktop-экранов 1440px: Dashboard, Model Detail, Data Drift Monitoring, Model Performance, Alerts and Incidents, Model Registry.
- Добавлены переиспользуемые UI-компоненты и стили: sidebar navigation, topbar controls, metric cards, status badges, alert cards, tables, charts, recommendations, registry actions.
- Макет заполнен реалистичными MLOps-данными для `churn_prediction_v3`, `fraud_detection_v2`, `demand_forecast_v1` и связанных сценариев качества моделей, drift, инцидентов и registry.
- Добавлен локальный Figma plugin generator, создающий редактируемые 1440px-фреймы MLOps dashboard внутри Figma при запуске в авторизованной Figma desktop/web-сессии.
- Добавлен подраздел в основной текст работы со сравнением трансформерного подхода с классическими системами ранжирования: TF-IDF, BM25, векторной моделью поиска, вероятностными моделями IR, PageRank и обучением ранжированию.
- Обновлен список литературы источниками по классической векторной модели поиска, TF-IDF/term specificity, PageRank и LambdaMART.
- Сравнительная таблица заменена на связный академический текст о роли классических методов и трансформерного подхода в архитектуре ATS.
