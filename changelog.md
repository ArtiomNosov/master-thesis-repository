# Changelog

## 2026-05-29

- Расширена русификация Reqcore: `ru.json` (settings, pipeline, onboarding, auth), `useLocalizedEnums`, `$t()` в SettingsSidebar и dashboard pipeline.
- Русский демо-seed: вакансии, кандидаты (ФИО), вопросы анкеты, заметки по откликам; org «Демо Reqcore».
- `RUNBOOK-RU.md`, доработан `russian-ui.client.ts` и `scripts/capture-i18n-screenshots.mjs` (localhost, UI-login fallback).
- Проверки: `npm run build`, `vitest` (361 тест), `db:reseed` — успешно.

## 2026-05-27

- Подключён git submodule `reqcore` → https://github.com/ArtiomNosov/reqcore, ветка `thesis/ru-localization`.
- В форке Reqcore: русская локаль `i18n/locales/ru.json`, язык по умолчанию `ru`, плагин `app/plugins/russian-ui.client.ts`, навигация через `$t('nav.*')`, убраны пользовательские формулировки с «ATS».
- Docker: Postgres + MinIO подняты, выполнены `db:migrate` и `db:seed`, проверен вход `demo@reqcore.com` / `demo1234` и активация org `reqcore-demo`.
- Скриншоты dashboard (22 страницы): `reqcore/artifacts/i18n-ru-screenshots/`, отчёт `dom-language-report.json`.
- Исправлен `russian-ui.client.ts` (больше не ломает гидратацию dashboard).

## 2026-05-25

- Добавлен расчет экономического обоснования внедрения в текст ВКР: исходные метрики времени и качества, денежные показатели, срок окупаемости и ROI.
- Обновлена презентация ВКР: добавлены слайды с базовыми метриками сравнения человека и модели, а также с денежными метриками внедрения.
