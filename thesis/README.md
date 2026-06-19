# Thesis LaTeX (PZ)

Единственный набор исходников: **`thesis/latex/`**.

## Сборка PDF

```powershell
# локально (MiKTeX / TeX Live)
.\thesis\build.ps1

# Docker (Windows / без локального TeX)
.\thesis\build.ps1 -Docker
```

Linux/macOS: `./thesis/build-docker.sh`

Выход: `thesis/latex/build/master-thesis-pz-body.pdf`

## Markdown → LaTeX

```powershell
py -3 thesis/scripts/md_to_latex_port.py
py -3 thesis/scripts/render_regex_figure.py   # при необходимости
py -3 thesis/scripts/render_flow_figure.py    # рис. 2 — flowchart Reqcore
```

Источник: `docs/obsidian/thesis/[USED] thesis_draft_final_ru.md`

## Прочее

| Путь | Назначение |
|------|------------|
| `latex/chapters/` | главы, преамбула, `biblio.bib` |
| `latex/figures/` | PDF-рисунки для `\includegraphics` |
| `scripts/` | порт MD → LaTeX, regex-figure |
| `upstream-template/` | git submodule — [шаблон кафедры](https://gitlab.com/skibcsit/thesis-template), только справочно |

```powershell
git submodule update --init thesis/upstream-template
```
