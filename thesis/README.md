# Thesis LaTeX (PZ)

Сборка пояснительной записки без титульных листов:

```powershell
.\thesis\build.ps1
```

## Структура

| Путь | Назначение |
|------|------------|
| `latex/` | Рабочие исходники: `master-thesis-pz-body.tex`, главы, `biblio.bib`, фигуры |
| `scripts/` | Markdown → LaTeX, фигуры, sync kit |
| `pz-latex-kit/` | Автономный kit для Docker-сборки |
| `upstream-template/` | **Git submodule** — [шаблон кафедры](https://gitlab.com/skibcsit/thesis-template) (только справочно) |

После клонирования репозитория:

```powershell
git submodule update --init thesis/upstream-template
```

Преамбула сборки: `latex/chapters/master-thesis-preamble.tex` (обрезанная версия upstream macro).

Markdown-источник: `docs/obsidian/thesis/[USED] thesis_draft_final_ru.md`
