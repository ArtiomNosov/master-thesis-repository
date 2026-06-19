$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$latexDir = Join-Path $PSScriptRoot 'latex'

$failures = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
$passes = New-Object System.Collections.Generic.List[string]

function Add-Pass($message) { [void]$passes.Add($message) }
function Add-Warn($message) { [void]$warnings.Add($message) }
function Add-Fail($message) { [void]$failures.Add($message) }

function Test-RequiredFile($relativePath) {
  $path = Join-Path $repoRoot $relativePath
  if (Test-Path -LiteralPath $path) {
    Add-Pass "found $relativePath"
  }
  else {
    Add-Fail "missing $relativePath"
  }
}

$requiredFiles = @(
  'thesis\latex\master-thesis-pz-body.tex',
  'thesis\latex\chapters\master-thesis-preamble.tex',
  'thesis\latex\.latexmkrc',
  'thesis\latex\figures\application_analysis_request_flow.pdf',
  'thesis\latex\figures\regex-section-headers.pdf',
  'thesis\upstream-template\chapters\thesis-template-macro.tex'
)

foreach ($file in $requiredFiles) {
  Test-RequiredFile $file
}

$texFiles = Get-ChildItem -LiteralPath $latexDir -Recurse -File -Include *.tex
$obsidianLinks = $texFiles | Select-String -Pattern '!\[\[' -SimpleMatch
if ($obsidianLinks) {
  foreach ($match in $obsidianLinks) {
    Add-Fail "Obsidian image/link syntax remains in $($match.Path):$($match.LineNumber)"
  }
}
else {
  Add-Pass 'no Obsidian image syntax found in LaTeX files'
}

$todoMatches = $texFiles | Select-String -Pattern 'TODO'
if ($todoMatches) {
  Add-Warn ("TODO markers remaining in LaTeX files: {0}" -f $todoMatches.Count)
}
else {
  Add-Pass 'no TODO markers found in LaTeX files'
}

$bib = Get-Content -Raw -LiteralPath (Join-Path $latexDir 'chapters\biblio.bib')
$bibEntries = ([regex]::Matches($bib, '@[A-Za-z]+\s*\{')).Count
if ($bibEntries -lt 30) {
  Add-Warn "bibliography has $bibEntries entries; final ВКР likely needs at least 30-35"
}
else {
  Add-Pass "bibliography entry count: $bibEntries"
}

if (Get-Command latexmk -ErrorAction SilentlyContinue) {
  Add-Pass 'latexmk is available locally'
}
else {
  Add-Warn 'latexmk is not available locally; use Docker or install MiKTeX/TeX Live'
}

if (Get-Command xelatex -ErrorAction SilentlyContinue) {
  Add-Pass 'xelatex is available locally'
}
else {
  Add-Warn 'xelatex is not available locally; use Docker or install MiKTeX/TeX Live'
}

Write-Host 'Thesis checks'
Write-Host '============='

foreach ($message in $passes) {
  Write-Host "[OK] $message" -ForegroundColor Green
}

foreach ($message in $warnings) {
  Write-Host "[WARN] $message" -ForegroundColor Yellow
}

foreach ($message in $failures) {
  Write-Host "[FAIL] $message" -ForegroundColor Red
}

if ($failures.Count -gt 0) {
  exit 1
}

exit 0
