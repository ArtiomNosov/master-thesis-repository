# E2E visual audit: render → XML validate → geometry → PNG → reports
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$AuditDir = "docs/obsidian/thesis/assets/_audit"
New-Item -ItemType Directory -Force -Path $AuditDir | Out-Null

Write-Host "==> Render + XML validate + layout geometry"
& "$PSScriptRoot\render_thesis_diagrams.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Export PNG previews"
py -3 scripts/export_thesis_diagram_png.py all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$LayoutJson = "$AuditDir/layout_report.json"
py -3 scripts/check_svg_layout.py --json $LayoutJson
$LayoutExit = $LASTEXITCODE

Write-Host ""
Write-Host "PNG previews:"
Get-ChildItem "$AuditDir\*.png" | ForEach-Object { Write-Host "  $($_.FullName)" }
Write-Host ""
Write-Host "Layout JSON: $LayoutJson (exit=$LayoutExit)"
Write-Host ""
Write-Host "Next: run vision agents (Cursor Task) with:"
Write-Host "  - $AuditDir/application_analysis_request_flow_marked.png"
Write-Host "  - $AuditDir/c4_container_architecture_marked.png"
Write-Host "  Prompt: .agents/prompts/diagram-visual-audit.md"
Write-Host "  Merge results into: $AuditDir/diagram_audit_report.md"

exit $LayoutExit
