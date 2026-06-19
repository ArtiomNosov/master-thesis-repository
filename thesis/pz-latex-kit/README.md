# PZ LaTeX build kit

Self-contained LaTeX sources for the **пояснительная записка** (abstract + main text + bibliography), without title pages or assignment.

## Contents

| Path | Purpose |
|------|---------|
| `master-thesis-pz-body.tex` | Root document |
| `chapters/` | Chapters, macros, bibliography |
| `figures/` | PDF figures (flowchart, regex) |
| `build/` | PDF output (generated) |

## Regenerate from Markdown (full repo)

From repository root:

```powershell
py -3 thesis/scripts/md_to_latex_port.py
py -3 thesis/scripts/render_regex_figure.py
py -3 thesis/scripts/sync_pz_latex_kit.py
```

Markdown source: `docs/obsidian/thesis/[USED] thesis_draft_final_ru.md`

Optional: re-render flowchart (requires Docker + Graphviz):

```powershell
docker run --rm -v "${PWD}/docs/obsidian/thesis/diagrams:/in" -v "${PWD}/thesis/latex/figures:/out" nshine/dot:latest dot -Tpdf -o /out/application_analysis_request_flow.pdf /in/application_analysis_request_flow.dot
```

## Build PDF (Docker, recommended on Windows)

From this directory (`thesis/pz-latex-kit`):

```powershell
docker run --rm -v "${PWD}:/work" -w /work aergus/latex:2022-01-02 bash -c "xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex && biber build/master-thesis-pz-body && xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex && xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex"
```

Output: `build/master-thesis-pz-body.pdf`

## Build PDF (Linux / TeX Live)

```bash
./build.sh
```

## Notes

- Engine: **XeLaTeX** + **biber**
- Cyrillic monospace in `\texttt{}` uses DejaVu Sans Mono in Docker (Linux) and Courier New on Windows
- API routes in Table 2.1 use `\path{...}` for correct `/` breaks
