# Changelog

## 2026-05-29

- E2E i18n: `npm run db:reseed`, dev на **http://localhost:3001** (3000 занят Langfuse), `dom-language-report.json` → `pagesWithUnexpectedLatin: []`, 24 скриншота обновлены.
- Исправлена загрузка локали `ru`: экранирован `@` в `emailPlaceholder` (`colleague{'@'}company.com`), иначе vue-i18n не компилировал `ru.json` и UI показывал ключи i18n.
- Расширена русификация Reqcore: `ru.json` (settings, pipeline, onboarding, auth), `useLocalizedEnums`, `$t()` в SettingsSidebar и dashboard pipeline.
- Русский демо-seed: вакансии, кандидаты (ФИО), вопросы анкеты, заметки по откликам; org «Демо Reqcore».
- `RUNBOOK-RU.md`, доработан `russian-ui.client.ts` и `scripts/capture-i18n-screenshots.mjs` (localhost, UI-login fallback, fail при видимых i18n-ключах).
- Проверки: `npm run build`, `vitest` (361 тест), `db:reseed` — успешно.

## 2026-05-27

- Рис. 2 (поток заявки): исправлена пустая легенда (убрана постобработка `move_legend_cluster_last`, легенда — HTML-таблица `sat_legend`); ветки горизонтально через `rank=same`, `splines=polyline`.
- Исправлена раскладка Graphviz: убраны фантомные узлы из имён кластеров в `rank=same`/рёбрах; легенда SAT закреплена снизу (`flow_anchor` / `metrics -> leg_assumption`).
- Раскладка Graphviz: `rankdir=TB`, горизонтальные цепочки в ветках (`rankdir=LR` в кластерах), легенда SAT в `cluster_legend`; `bi_gate`/`llm_gate` — рамка «Ветка bi-encoder» охватывает «Вызов bi-encoder сервиса».
- Скрипт `scripts/render_thesis_diagrams.ps1` и постобработка SVG (легенда в конце DOM).
- Рис. 1 (C4): источник [`c4_container_architecture.dot`](docs/obsidian/thesis/diagrams/c4_container_architecture.dot), рендер в `c4_container_architecture_marked.svg` (вместо ручного SVG).
- `py -3 scripts/render_thesis_diagram.py all` — пересборка обеих thesis-диаграмм.
- Рис. 2 (поток анализа заявки): источник Graphviz [`application_analysis_request_flow.dot`](docs/obsidian/thesis/diagrams/application_analysis_request_flow.dot), рендер `py -3 scripts/render_thesis_diagram.py`; SVG пересобирается автоматически, SAT — цветом рамки + `cluster_legend`.
- Справочник цветов SAT: [`sat_styles.inc`](docs/obsidian/thesis/diagrams/sat_styles.inc); инструкция: [`diagrams/README.ru.md`](docs/obsidian/thesis/diagrams/README.ru.md).
- Валидатор SVG: отдельные правила для Graphviz (без проверки координат текста) и BPMN overlay.
- Влит актуальный `semester-4` в `cursor/separate-architecture-results-771d` для автоматического merge PR.
- Переработана вёрстка диаграмм C4 и потока анализа (SAT): статус кодируется цветом рамки.
- Исправлены BPMN AS IS / TO BE: устранены наложения подписей.
- В текст ВКР добавлена нотация SAT; маркеры заменены на органичные формулировки.
- Скрипты `scripts/validate_thesis_diagram_svgs.py`, `scripts/organic_thesis_markers.py`.
- Подключён submodule `reqcore`, русификация UI, Docker/E2E проверка, скриншоты i18n.
- Нейтрализованы упоминания SuperJob; термины «независимый энкодер (bi-encoder)».
- Термины приведены к виду «по-русски (идентификатор)» в диаграммах и API-разделе.
- Слияние сравнительного подраздела, MLOps-макет, baseline-метрики в DOCX/презентации.
- В таблице задачи 1 убран дублирующий столбец «Комментарий»; скрипт агента перенесён в `.agents/artifacts/patch_vkr_texts.py`.
- В сравнительном подразделе: русские формулировки вместо «эмбеддинг» и сокращений LTR; полные названия вместо BERT/Sentence-BERT/bi-encoder.

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

- Единообразные маркеры SAT в тексте диссертации; уточнены формулировки API, Reqcore, экономики, экспериментов.
- SVG-диаграммы C4, потока анализа, AS IS / TO BE с визуальной легендой SAT.
- High-fidelity HTML-макет MLOps-мониторинга, Figma plugin, PNG-экспорты.
- Сравнительный подраздел: TF-IDF, BM25, PageRank, LambdaMART vs трансформерный подход.
- Расчёт экономического обоснования внедрения; слайды baseline и денежных метрик в презентации.


