param(
  [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$latexDir = Join-Path $PSScriptRoot 'latex'
$entrypoint = 'master-thesis-pz-body.tex'

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
  Write-Host 'Or build with Docker — see thesis/pz-latex-kit/README.md'
  exit 2
}

Push-Location $latexDir
try {
  if ($Clean) {
    latexmk -C $entrypoint
  }

  latexmk -xelatex -interaction=nonstopmode -halt-on-error $entrypoint

  $pdf = Join-Path $latexDir 'build\master-thesis-pz-body.pdf'
  if (-not (Test-Path -LiteralPath $pdf)) {
    Write-Error "Build finished but expected PDF was not found: $pdf"
    exit 1
  }

  Write-Host "PDF built: $pdf" -ForegroundColor Green
}
finally {
  Pop-Location
}
