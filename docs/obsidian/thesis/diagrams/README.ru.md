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

1. **`rankdir=TB`** на корне — схема сверху вниз, **легенда SAT снизу** (`subgraph cluster_legend`, якорь `flow_anchor` / ребро от `metrics`).
2. **`subgraph cluster_flow`** с **`rankdir=LR`** в позвоночнике и ветках.
3. **Ветки** — вход через **`bi_gate` / `llm_gate`** (`shape=point`), ребро `provider -> bi_gate` с `lhead=cluster_biencoder`.
4. **Не использовать имена кластеров как узлы** в `{ rank=same; ... }` или рёбрах — Graphviz создаёт «фантомные» блоки и ломает раскладку.

## SAT (Status Assertion Taxonomy)

Справочник цветов: [`sat_styles.inc`](sat_styles.inc). В `.dot` на каждом узле задаются `fillcolor` и `color`. Статус **только цветом рамки**, без дублирующих подписей внутри блоков.

## Файлы

| Файл | Назначение |
|------|------------|
| `application_analysis_request_flow.dot` | Рис. 2 — flowchart Reqcore / bi-encoder / LLM |
| `c4_container_architecture.dot` | Рис. 1 — C4 Container Reqcore + ML |
| `sat_styles.inc` | Справочник SAT-цветов |

BPMN (`archive/.../as_is.svg`, `to_be.svg`) пока не в Graphviz-пайплайне.
