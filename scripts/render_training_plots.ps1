param(
    [string]$Python = "python",
    [string]$InputCsv = "checkpoints/model/eval/binary_classification_evaluation_val-binary_results.csv",
    [string]$Metric = "cosine_f1",
    [string]$Output = "docs/obsidian/thesis/assets/biencoder_cosine_f1_validation_ru.png",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $Root
try {
    if (-not $SkipInstall) {
        & $Python -m pip install -r "experiments/requirements-plotting.txt"
    }

    & $Python "experiments/scripts/19_plot_training_metric.py" `
        --input $InputCsv `
        --metric $Metric `
        --output $Output
}
finally {
    Pop-Location
}
