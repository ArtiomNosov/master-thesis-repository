# Диаграммы ВКР (исходники)

Рисунки в тексте ссылаются на SVG в [`../assets/`](../assets/). **Источник правды** — файлы `.dot` в этой папке; SVG пересобираются скриптом, вручную координаты в SVG не правим.

## Зависимость: Graphviz

Нужен `dot` в PATH.

**Windows:**

```powershell
winget install Graphviz.Graphviz
```

Проверка: `dot -V`

## Пересборка

Из корня репозитория:

```powershell
.\scripts\render_thesis_diagrams.ps1
```

или:

```powershell
py -3 scripts/render_thesis_diagram.py all
py -3 scripts/validate_thesis_diagram_svgs.py
```

По одной диаграмме:

```powershell
py -3 scripts/render_thesis_diagram.py application_analysis_request_flow
py -3 scripts/render_thesis_diagram.py c4_container_architecture
```

## Шаблон раскладки

1. **`rankdir=TB`** + **`rank=same`** на каждом ряду — горизонтальные цепочки (позвоночник, ветки); **легенда** — узел `sat_legend` (HTML-таблица), якорь `flow_anchor`.
2. **`splines=polyline`** (не `ortho`) — читаемые рёбра между рядами.
3. **Ветки** — `bi_gate` / `llm_gate` (`shape=point`), `provider -> bi_gate` с `lhead=cluster_biencoder`.
4. **Не использовать имена кластеров как узлы** и не переносить в SVG только рамку `cluster_legend` — иначе легенда визуально «пустая».

## SAT (Status Assertion Taxonomy)

Справочник цветов: [`sat_styles.inc`](sat_styles.inc). В `.dot` на каждом узле задаются `fillcolor` и `color`. Статус **только цветом рамки**, без дублирующих подписей внутри блоков.

## Файлы

| Файл | Назначение |
|------|------------|
| `application_analysis_request_flow.dot` | Рис. 2 — flowchart Reqcore / bi-encoder / LLM |
| `c4_container_architecture.dot` | Рис. 1 — C4 Container Reqcore + ML |
| `sat_styles.inc` | Справочник SAT-цветов |

BPMN (`archive/.../as_is.svg`, `to_be.svg`) пока не в Graphviz-пайплайне.

## E2E визуальная проверка

```powershell
.\scripts\run_diagram_visual_e2e.ps1
```

Создаёт PNG в [`../assets/_audit/`](../assets/_audit/), `layout_report.json`, затем vision-агенты по чеклисту [`.agents/prompts/diagram-visual-audit.md`](../../../.agents/prompts/diagram-visual-audit.md).
