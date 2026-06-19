# LaTeX PZ

Entrypoint: `master-thesis-pz-body.tex`

Сборка — из корня репозитория:

```powershell
..\build.ps1          # локальный TeX
..\build.ps1 -Docker  # Docker (aergus/latex:2022-01-02)
```

Порт из Markdown: `../scripts/md_to_latex_port.py`

Upstream (submodule): `../upstream-template/`
