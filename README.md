# Master Thesis Repository

Репозиторий для магистерской дипломной работы.

## Reqcore (submodule)

Интерфейс системы подбора — отдельный форк в submodule:

- Репозиторий: https://github.com/ArtiomNosov/reqcore
- Ветка: `thesis/ru-localization` (русский UI по умолчанию)
- Путь в этом репозитории: `reqcore/`

```bash
git submodule update --init --recursive
cd reqcore
git checkout thesis/ru-localization
npm ci
cp .env.example .env   # или ./setup.sh
docker compose up -d
npm run dev
```

Локальный URL: http://localhost:3000

Обновить submodule до последнего коммита ветки:

```bash
cd reqcore
git pull origin thesis/ru-localization
cd ..
git add reqcore
```

## Структура репозитория

- `reqcore/`: Submodule — UI Reqcore (Nuxt), русская локализация.
- `docs/`: Текст диплома (.tex, .md, презентации) и скомпилированные PDF.
- `code/`: Основной переиспользуемый код (алгоритмы, классы, утилиты).
- `data/`: Датасеты и загруженные файлы (игнорируются git).
- `experiments/`: Скрипты для запуска, логи тренировок и ноутбуки (.ipynb).
- `docs/references/`: Статьи в PDF (литобзор) и .bib файлы.
- `prompts/`: Набор промптов для работы с ИИ ассистентами.
- `archive/`: Архивные материалы из прошлых семестров.
