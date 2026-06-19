# LaTeX build (PZ body)

Entrypoint: `master-thesis-pz-body.tex` — реферат, основной текст и список литературы без титульных листов.

```powershell
# from repo root
.\thesis\build.ps1

# or Docker (see thesis/pz-latex-kit/README.md)
```

Upstream department template (reference only): `thesis/_upstream-thesis-template/`.

Markdown → LaTeX: `thesis/scripts/md_to_latex_port.py`
