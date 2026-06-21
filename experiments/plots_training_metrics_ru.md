# Воспроизводимое построение графика обучения

Источник данных для графика F1-меры:

```text
checkpoints/model/eval/binary_classification_evaluation_val-binary_results.csv
```

Команда полного воспроизведения PNG с русскими подписями:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/render_training_plots.ps1
```

Команда без повторной установки зависимостей:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/render_training_plots.ps1 -SkipInstall
```

Выходной файл по умолчанию:

```text
docs/obsidian/thesis/assets/biencoder_cosine_f1_validation_ru.png
```

Для другой метрики можно передать имя столбца CSV и путь результата:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/render_training_plots.ps1 `
  -Metric cosine_precision `
  -Output docs/obsidian/thesis/assets/biencoder_cosine_precision_validation_ru.png
```
