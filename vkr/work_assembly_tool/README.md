# VКР work assembly tool

Инструментальная подпапка для сборки текста ВКР из Markdown в Word DOCX.

## Состав

- `docs/vkr_formatting_rules.md` -- извлеченные правила оформления ВКР из документов кафедры.
- `docs/markdown_reference.md` -- краткая техническая справка по Markdown/CommonMark для входного формата.
- `docs/word_docx_reference.md` -- краткая техническая справка по Word DOCX и `python-docx`.
- `vkr_format_config.json` -- параметризованные правила, которые применяются конвертером и чекером.
- `scripts/md_to_vkr_docx.py` -- конвертер Markdown -> DOCX.
- `scripts/setup_plantuml.py` -- скачивание `plantuml.jar` для локального рендера диаграмм.
- `scripts/start_plantuml_server.py` -- опциональный локальный HTTP-сервер PlantUML (режим `server`).
- `scripts/check_docx_format.py` -- структурная проверка DOCX по JSON-правилам.
- `examples/sample_vkr_fragment.md` -- небольшой тестовый фрагмент Markdown.
- `sources/` -- локальные копии и извлеченный текст исходных регламентов ВКР.

## Быстрый запуск

```bash
python3 -m pip install --user -r vkr/work_assembly_tool/requirements.txt
python3 vkr/work_assembly_tool/scripts/setup_plantuml.py
python3 vkr/work_assembly_tool/scripts/md_to_vkr_docx.py \
  --input docs/obsidian/thesis/[USED]\ thesis_draft_final_ru.md \
  --config vkr/work_assembly_tool/vkr_format_config.json \
  --output vkr/work_assembly_tool/out/thesis_draft_final_ru.docx
python3 vkr/work_assembly_tool/scripts/check_docx_format.py \
  --docx vkr/work_assembly_tool/out/thesis_draft_final_ru.docx \
  --config vkr/work_assembly_tool/vkr_format_config.json
```

### PlantUML (локально, без plantuml.com)

По умолчанию диаграммы рендерятся через локальный `java -jar tools/plantuml.jar -tpng -pipe` (режим `local_jar` в `vkr_format_config.json`). JAR скачивается один раз:

```bash
py vkr/work_assembly_tool/scripts/setup_plantuml.py
```

Требуется установленная Java (JRE/JDK). JAR берётся с [plantuml.com/download](https://plantuml.com/download) (GitHub release `plantuml-1.2026.5.jar`).

Опционально — локальный HTTP-сервер (режим `server`):

```bash
py vkr/work_assembly_tool/scripts/start_plantuml_server.py
```

В конфиге: `"plantuml_mode": "server"`, `"plantuml_server_url": "http://127.0.0.1:8080/png/"`.

После открытия DOCX в Word нажмите **F9** на странице «Содержание», чтобы обновить оглавление.

Проверка форматирования выполняется программно по структуре DOCX; визуальная оценка не используется. Чекер падает, если таблица не имеет подписи над таблицей, рисунок не имеет подписи под рисунком или любой paragraph/character style либо текстовый run имеет нечерный цвет. Если в Markdown перед таблицей нет подписи `Таблица N – ...`, конвертер добавляет ее автоматически на основе заголовков таблицы.
