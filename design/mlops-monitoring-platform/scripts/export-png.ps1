$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutDir = Join-Path $RootDir "exports\png"
$ChromeBin = $env:CHROME_BIN

if (-not $ChromeBin) {
  $candidates = @(
    "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "${env:LOCALAPPDATA}\Google\Chrome\Application\chrome.exe"
  )
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      $ChromeBin = $candidate
      break
    }
  }
}

if (-not $ChromeBin) {
  throw "Chrome не найден. Укажите путь через переменную CHROME_BIN."
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$screens = @(
  @{ screen = "dashboard"; file = "01-main-mlops-dashboard.png" },
  @{ screen = "model-detail"; file = "02-model-detail.png" },
  @{ screen = "data-drift"; file = "03-data-drift-monitoring.png" },
  @{ screen = "model-performance"; file = "04-model-performance.png" },
  @{ screen = "alerts-incidents"; file = "05-alerts-incidents.png" },
  @{ screen = "model-registry"; file = "06-model-registry.png" }
)

foreach ($item in $screens) {
  $outFile = Join-Path $OutDir $item.file
  $profileDir = Join-Path $env:TEMP ("mlops-export-" + [guid]::NewGuid().ToString())
  New-Item -ItemType Directory -Force -Path $profileDir | Out-Null

  if (Test-Path $outFile) {
    Remove-Item $outFile -Force
  }

  $url = "file:///$($RootDir.Replace('\', '/'))/index.html?screen=$($item.screen)"
  & $ChromeBin `
    --headless=new `
    --disable-gpu `
    --hide-scrollbars `
    --user-data-dir="$profileDir" `
    --window-size=1440,1024 `
    --screenshot="$outFile" `
    $url | Out-Null

  Remove-Item $profileDir -Recurse -Force

  if (-not (Test-Path $outFile) -or (Get-Item $outFile).Length -eq 0) {
    throw "Не удалось экспортировать экран $($item.screen) в $outFile"
  }
}

Write-Host "PNG-экспорты сохранены в $OutDir"
