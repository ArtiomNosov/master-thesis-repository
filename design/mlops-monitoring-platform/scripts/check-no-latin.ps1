$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$HtmlPath = Join-Path $RootDir "index.html"
$Html = Get-Content -Path $HtmlPath -Raw -Encoding UTF8

if ($Html -match '(?s)<body>(.*)</body>') {
  $Body = $Matches[1]
} else {
  throw "Не найден блок body в $HtmlPath"
}

$Body = $Body -replace '(?s)<script.*?</script>', ''
$Text = [regex]::Replace($Body, '<[^>]+>', ' ')
$Text = [regex]::Replace($Text, '\s+', ' ').Trim()

$Matches = [regex]::Matches($Text, '[A-Za-z]+')
if ($Matches.Count -gt 0) {
  Write-Host "Найдена латиница в видимом тексте index.html:" -ForegroundColor Red
  $Matches | ForEach-Object { $_.Value } | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" }
  exit 1
}

Write-Host "Проверка пройдена: латиница в видимом тексте index.html не найдена."
