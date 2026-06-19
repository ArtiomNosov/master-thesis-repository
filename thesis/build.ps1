param(
  [switch]$Clean,
  [switch]$Docker
)

$ErrorActionPreference = 'Stop'

$latexDir = Join-Path $PSScriptRoot 'latex'
$entrypoint = 'master-thesis-pz-body.tex'
$entryPath = Join-Path $latexDir $entrypoint

if (-not (Test-Path -LiteralPath $entryPath)) {
  Write-Error "Missing LaTeX entrypoint: thesis/latex/$entrypoint"
  exit 1
}

function Invoke-DockerBuild {
  $mount = (Resolve-Path -LiteralPath $latexDir).Path -replace '\\', '/'
  if ($Clean) {
    docker run --rm -v "${mount}:/work" -w /work aergus/latex:2022-01-02 bash -c "rm -rf build/master-thesis-pz-body.*"
  }
  docker run --rm -v "${mount}:/work" -w /work aergus/latex:2022-01-02 bash -c @"
mkdir -p build &&
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex &&
biber build/master-thesis-pz-body &&
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex &&
xelatex -interaction=nonstopmode -output-directory=build master-thesis-pz-body.tex
"@
  $pdf = Join-Path $latexDir 'build\master-thesis-pz-body.pdf'
  if (-not (Test-Path -LiteralPath $pdf)) {
    Write-Error "Build finished but expected PDF was not found: $pdf"
    exit 1
  }
  Write-Host "PDF built: $pdf" -ForegroundColor Green
}

if ($Docker) {
  Invoke-DockerBuild
  exit 0
}

$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
$xelatex = Get-Command xelatex -ErrorAction SilentlyContinue

if (-not $latexmk -or -not $xelatex) {
  Write-Host 'Local LaTeX toolchain not found. Use Docker:' -ForegroundColor Yellow
  Write-Host '  .\thesis\build.ps1 -Docker' -ForegroundColor Yellow
  exit 2
}

Push-Location $latexDir
try {
  New-Item -ItemType Directory -Force -Path (Join-Path $latexDir 'build') | Out-Null

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
