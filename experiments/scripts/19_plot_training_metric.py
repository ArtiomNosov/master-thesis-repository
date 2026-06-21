from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "checkpoints"
    / "model"
    / "eval"
    / "binary_classification_evaluation_val-binary_results.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "obsidian"
    / "thesis"
    / "assets"
    / "biencoder_cosine_f1_validation_ru.png"
)

METRIC_LABELS = {
    "cosine_accuracy": "Доля правильных ответов",
    "cosine_f1": "F1-мера по косинусному сходству",
    "cosine_precision": "Точность по косинусному сходству",
    "cosine_recall": "Полнота по косинусному сходству",
    "cosine_ap": "Средняя точность по косинусному сходству",
    "cosine_mcc": "Коэффициент корреляции Мэтьюса",
}

METRIC_TITLES = {
    "cosine_accuracy": "Динамика доли правильных ответов на валидационной выборке",
    "cosine_f1": "Динамика F1-меры по косинусному сходству на валидационной выборке",
    "cosine_precision": "Динамика точности по косинусному сходству на валидационной выборке",
    "cosine_recall": "Динамика полноты по косинусному сходству на валидационной выборке",
    "cosine_ap": "Динамика средней точности по косинусному сходству на валидационной выборке",
    "cosine_mcc": "Динамика коэффициента корреляции Мэтьюса на валидационной выборке",
}


def read_metric_points(input_csv: Path, metric: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []

    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {input_csv}")
        if "steps" not in reader.fieldnames:
            raise ValueError(f"CSV file has no 'steps' column: {input_csv}")
        if metric not in reader.fieldnames:
            raise ValueError(f"CSV file has no '{metric}' column: {input_csv}")

        for row_number, row in enumerate(reader, start=2):
            raw_step = row.get("steps")
            raw_value = row.get(metric)
            if raw_step in (None, "") or raw_value in (None, ""):
                continue
            try:
                steps.append(int(float(raw_step)))
                values.append(float(raw_value))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in {input_csv} at row {row_number}"
                ) from exc

    if not steps:
        raise ValueError(f"No points for metric '{metric}' in {input_csv}")

    return steps, values


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 150,
        }
    )

    return plt, FuncFormatter


def render_plot(
    steps: list[int],
    values: list[float],
    metric: str,
    output: Path,
    title: str | None,
    ylabel: str | None,
) -> None:
    plt, FuncFormatter = configure_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        steps,
        values,
        color="#1f77b4",
        marker="o",
        markersize=7,
        linewidth=2.2,
    )

    ax.set_title(title or METRIC_TITLES.get(metric, f"Динамика метрики {metric}"))
    ax.set_xlabel("Шаг обучения")
    ax.set_ylabel(ylabel or METRIC_LABELS.get(metric, metric))

    ax.grid(True, alpha=0.35)
    ax.set_xlim(min(steps) - 75, max(steps) + 75)

    min_value = min(values)
    max_value = max(values)
    padding = max((max_value - min_value) * 0.08, 0.01)
    ax.set_ylim(max(0.0, min_value - padding), min(1.0, max_value + padding))

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value):,}".replace(",", " "))
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", ","))
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Russian-labeled training metric plot from an evaluation CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to SentenceTransformers binary classification evaluation CSV.",
    )
    parser.add_argument(
        "--metric",
        default="cosine_f1",
        help="Metric column to render, for example cosine_f1 or cosine_ap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    parser.add_argument("--title", help="Optional Russian plot title.")
    parser.add_argument("--ylabel", help="Optional Russian Y-axis label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps, values = read_metric_points(args.input, args.metric)
    render_plot(steps, values, args.metric, args.output, args.title, args.ylabel)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
