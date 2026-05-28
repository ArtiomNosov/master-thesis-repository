#!/usr/bin/env python3
"""One-off agent helper: patch VKR docx/pptx (not part of thesis deliverables)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "vkr" / "artifacts"
BASELINE_REF = "origin/cursor/objective-baseline-comparison-35bc"

COMMON_REPLACEMENTS = [
    ("агрегатора SuperJob", "агрегатора вакансий"),
    ("из агрегатора SuperJob", "из агрегатора вакансий"),
    ("данных SuperJob", "данных агрегатора вакансий"),
    ("SuperJob", "агрегатор вакансий"),
    ("data/super_job", "каталога данных агрегатора"),
    ("DCNG@", "NDCG@"),
]

DOCX_REPLACEMENTS = COMMON_REPLACEMENTS + [
    ("би-энкодера", "независимого энкодера"),
    ("би-энкодере", "независимом энкодере"),
    ("би-энкодеру", "независимому энкодеру"),
    ("би-энкодером", "независимым энкодером"),
    ("би-энкодер", "независимый энкодер"),
    ("bi-encoder архитектуры", "архитектуры независимого энкодера (bi-encoder)"),
    ("bi-encoder подхода", "подхода независимого энкодера (bi-encoder)"),
    ("bi-encoder модель", "модель независимого энкодера (bi-encoder)"),
    ("bi-encoder модели", "модели независимого энкодера (bi-encoder)"),
    ("bi-encoder путь", "путь независимого энкодера (bi-encoder)"),
    ("bi-encoder оценки", "оценки независимым энкодером (bi-encoder)"),
    ("bi-encoder scoring", "скоринга независимым энкодером (bi-encoder)"),
    ("bi-encoder сервис", "сервис независимого энкодера (bi-encoder)"),
    ("bi-encoder микросервисом", "микросервисом независимого энкодера (bi-encoder)"),
    ("bi-encoder ranking request", "запроса ранжирования независимым энкодером (bi-encoder)"),
    ("bi-encoder зависит", "стоимость независимого энкодера (bi-encoder) зависит"),
    ("для bi-encoder", "для независимого энкодера (bi-encoder)"),
    ("Для bi-encoder", "Для независимого энкодера (bi-encoder)"),
    ("Если bi-encoder", "Если независимый энкодер (bi-encoder)"),
    ("bi-encoder как", "независимого энкодера (bi-encoder) как"),
    ("bi-encoder в разделе", "независимого энкодера (bi-encoder) в разделе"),
    ("выбор bi-encoder", "выбор архитектуры независимого энкодера (bi-encoder)"),
    ("использования bi-encoder", "использования независимого энкодера (bi-encoder)"),
    ("адаптации bi-encoder", "адаптации независимого энкодера (bi-encoder)"),
    ("ограничений bi-encoder", "ограничений независимого энкодера (bi-encoder)"),
    ("применения bi-encoder", "применения независимого энкодера (bi-encoder)"),
    ("bi-encoder архитектура", "архитектура независимого энкодера (bi-encoder)"),
]

PPTX_REPLACEMENTS = COMMON_REPLACEMENTS + [
    ("би-энкодера", "независимого энкодера"),
    ("би-энкодере", "независимом энкодере"),
    ("би-энкодером", "независимым энкодером"),
    ("би-энкодер", "независимый энкодер"),
    ("Bi-Encoder (Sentence-BERT)", "независимый энкодер (bi-encoder, Sentence-BERT)"),
    ("Bi-Encoder", "независимый энкодер (bi-encoder)"),
    ("bi-encoder", "независимый энкодер (bi-encoder)"),
    ("<a:t>Комментарий</a:t>", ""),
    ("<a:t>val-выборка</a:t>", ""),
    ("<a:t>+281%</a:t>", ""),
    ("<a:t>Прирост</a:t>", ""),
    ("<a:t>Baseline</a:t>", "<a:t>Базовая модель</a:t>"),
    ("<a:t>Fine-Tuned</a:t>", "<a:t>Дообученная модель</a:t>"),
]

MARKDOWN_FILES = [
    ROOT / "docs/obsidian/presentation_defense.md",
    ROOT / "docs/obsidian/presentation_plan.md",
    ROOT / "docs/obsidian/thesis/[USED] thesis_draft_final_ru.md",
    ROOT / "docs/obsidian/Data_Preparation_and_Profiling_Plan.md",
    ROOT / "docs/obsidian/Data_Preparation_Final_Report.md",
    ROOT / "docs/obsidian/Baseline_Comparison_Report.md",
    ROOT / "changelog.md",
]


def git_show_blob(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def patch_zip_bytes(data: bytes, xml_predicate, replacements: list[tuple[str, str]]) -> bytes:
    src = zipfile.ZipFile(BytesIO(data))
    out_buf = BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for info in src.infolist():
            payload = src.read(info.filename)
            if xml_predicate(info.filename):
                text = payload.decode("utf-8")
                text = apply_replacements(text, replacements)
                payload = text.encode("utf-8")
            out.writestr(info, payload)
    return out_buf.getvalue()


def artifact_path(suffix: str) -> Path:
    skip = ("задание", "отзыв")
    matches = [
        path
        for path in ARTIFACTS.iterdir()
        if path.name.endswith(suffix) and not any(token in path.name for token in skip)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one artifact *{suffix}, found: {matches}")
    return matches[0]


def restore_baseline_artifacts() -> None:
    for rel in [
        "vkr/artifacts/М24-534_НосовАИ_ВКР.docx",
        "vkr/artifacts/М24-534_НосовАИ_презентация_ВКР.pptx",
    ]:
        (ROOT / rel).write_bytes(git_show_blob(BASELINE_REF, rel))


def patch_markdown() -> None:
    md_replacements = DOCX_REPLACEMENTS + [
        ("| Метрика | Baseline | Fine-Tuned | Прирост |", "| Метрика | Базовая модель | Дообученная модель |"),
        ("| Average Precision | 0.259 | **0.987** | **+281%** |", "| Average Precision | 0.259 | **0.987** |"),
        ("| F1-score | — | **0.960** | — |", "| F1-score | 0.508 | **0.960** |"),
        ("| Accuracy | — | **0.973** | — |", "| Accuracy | 0.660 | **0.973** |"),
        ("| **Bi-Encoder (Sentence-BERT)** |", "| **независимый энкодер (bi-encoder, Sentence-BERT)** |"),
        ("**Bi-Encoder (Sentence-BERT)**", "**независимый энкодер (bi-encoder, Sentence-BERT)**"),
        ("локальной bi-encoder модель", "локальной модели независимого энкодера (bi-encoder)"),
        ("(super_job)", "(каталог данных агрегатора)"),
        ("super_job", "каталога данных агрегатора"),
        ("#superjob", ""),
        ("практически случайная", "ниже качества дообученной модели на той же val-выборке"),
        ("это скачок почти в четыре раза", "это существенный рост относительно базовой модели на той же val-выборке"),
        ("вырос почти в четыре раза", "вырос относительно базовой модели на val-выборке"),
    ]
    for path in MARKDOWN_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text2 = apply_replacements(text, md_replacements)
        if text2 != text:
            path.write_text(text2, encoding="utf-8")
            print("Updated markdown:", path.relative_to(ROOT))


def main() -> int:
    restore_baseline_artifacts()

    docx = artifact_path("_ВКР.docx")
    pptx = artifact_path("_презентация_ВКР.pptx")

    docx.write_bytes(
        patch_zip_bytes(
            docx.read_bytes(),
            lambda n: n.endswith(".xml") and (n.startswith("word/") or n.startswith("docProps/")),
            DOCX_REPLACEMENTS,
        )
    )
    pptx.write_bytes(
        patch_zip_bytes(
            pptx.read_bytes(),
            lambda n: n.endswith(".xml") and n.startswith("ppt/"),
            PPTX_REPLACEMENTS,
        )
    )

    patch_markdown()
    print("Patched:", docx.name, pptx.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
