param(
  [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$latexDir = Join-Path $PSScriptRoot 'latex'
$entrypoint = 'master-thesis-3-pz.tex'

if (-not (Test-Path -LiteralPath (Join-Path $latexDir $entrypoint))) {
  Write-Error "Missing LaTeX entrypoint: thesis/latex/$entrypoint"
  exit 1
}

$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
$xelatex = Get-Command xelatex -ErrorAction SilentlyContinue

if (-not $latexmk -or -not $xelatex) {
  Write-Host 'LaTeX toolchain is not installed in this environment.' -ForegroundColor Yellow
  Write-Host 'Required commands: latexmk and xelatex.' -ForegroundColor Yellow
  Write-Host ''
  Write-Host 'Install one of the following, then rerun thesis/build.ps1:'
  Write-Host '- MiKTeX for Windows: https://miktex.org/download'
  Write-Host '- TeX Live: https://tug.org/texlive/'
  Write-Host ''
  Write-Host 'CI is configured in .github/workflows/build-thesis.yml and uses the upstream template image.'
  exit 2
}

Push-Location $latexDir
try {
  if ($Clean) {
    latexmk -C $entrypoint
  }

  latexmk -xelatex -interaction=nonstopmode -halt-on-error $entrypoint

  $pdf = Join-Path $latexDir 'build\master-thesis-3-pz.pdf'
  if (-not (Test-Path -LiteralPath $pdf)) {
    Write-Error "Build finished but expected PDF was not found: $pdf"
    exit 1
  }

  Write-Host "PDF built: $pdf" -ForegroundColor Green
}
finally {
  Pop-Location
}
