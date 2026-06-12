# Пересборка всех Graphviz-диаграмм ВКР (рис. 1 и рис. 2)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
py -3 scripts/render_thesis_diagram.py all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -3 scripts/validate_thesis_diagram_svgs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -3 scripts/check_svg_layout.py
exit $LASTEXITCODE
