# Thesis Workspace

This folder contains the working ВКР layer for the repository.

- `latex/` - local snapshot of the department LaTeX template plus the current thesis draft.
- `forms/` - place official title/task/signature PDFs here before final submission.
- `presentation/` - department 16:9 presentation template and notes.
- `references/` - provided reference PDFs and methodology files.
- `build.ps1`, `check.ps1` - repository-level helper scripts for build and structural checks.

Primary LaTeX entrypoint:

```powershell
.\thesis\build.ps1
```

Local TeX is not installed in this environment at the time of setup. The scripts are designed to fail with actionable instructions instead of silently producing partial output.
