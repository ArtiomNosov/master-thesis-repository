# VКР work assembly tool

Инструментальная подпапка для сборки текста ВКР из Markdown в Word DOCX.

## Состав

- `docs/vkr_formatting_rules.md` -- извлеченные правила оформления ВКР из документов кафедры.
- `docs/markdown_reference.md` -- краткая техническая справка по Markdown/CommonMark для входного формата.
- `docs/word_docx_reference.md` -- краткая техническая справка по Word DOCX и `python-docx`.
- `vkr_format_config.json` -- параметризованные правила, которые применяются конвертером и чекером.
- `scripts/md_to_vkr_docx.py` -- конвертер Markdown -> DOCX.
- `scripts/check_docx_format.py` -- структурная проверка DOCX по JSON-правилам.
- `examples/sample_vkr_fragment.md` -- небольшой тестовый фрагмент Markdown.
- `sources/` -- локальные копии и извлеченный текст исходных регламентов ВКР.

## Быстрый запуск

```bash
python3 -m pip install --user -r vkr/work_assembly_tool/requirements.txt
python3 vkr/work_assembly_tool/scripts/md_to_vkr_docx.py \
  --input docs/obsidian/thesis/[USED]\ thesis_draft_final_ru.md \
  --config vkr/work_assembly_tool/vkr_format_config.json \
  --output vkr/work_assembly_tool/out/thesis_draft_final_ru.docx
python3 vkr/work_assembly_tool/scripts/check_docx_format.py \
  --docx vkr/work_assembly_tool/out/thesis_draft_final_ru.docx \
  --config vkr/work_assembly_tool/vkr_format_config.json
```

Проверка форматирования выполняется программно по структуре DOCX; визуальная оценка не используется.
