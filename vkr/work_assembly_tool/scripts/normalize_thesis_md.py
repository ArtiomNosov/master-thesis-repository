#!/usr/bin/env python3
"""Normalize thesis markdown headings, captions and spacing for DOCX conversion."""

from __future__ import annotations

import re
from pathlib import Path

THESIS = Path("docs/obsidian/thesis/[USED] thesis_draft_final_ru.md")

HEADING_MAP = {
    "**Системный анализ предметной области рекрутмента, процессов найма и ATS-систем**": "**1.1. Системный анализ предметной области рекрутмента, процессов найма и ATS-систем**",
    "**Аналитический обзор методов обработки естественного языка (NLP) и алгоритмов машинного обучения (Learning to Rank)**": "**1.2. Аналитический обзор методов обработки естественного языка (NLP) и алгоритмов машинного обучения (Learning to Rank)**",
    "**Обоснование выбора архитектуры нейросетевых моделей (BERT, Sentence-BERT) и методов векторизации**": "**1.3. Обоснование выбора архитектуры нейросетевых моделей (BERT, Sentence-BERT) и методов векторизации**",
    "**Выполнение сбора, профилирования, очистки и разметки исходных данных (вакансии и резюме)**": "**2.1. Выполнение сбора, профилирования, очистки и разметки исходных данных (вакансии и резюме)**",
    "**Проектирование архитектуры пайплайна предварительной обработки и модуля семантического анализа**": "**2.2. Проектирование архитектуры пайплайна предварительной обработки и модуля семантического анализа**",
    "**Спецификация программного интерфейса (API) и модулей интеграции с ядром ATS**": "**2.3. Спецификация программного интерфейса (API) и модулей интеграции с ядром ATS**",
    "**Интеграция Reqcore с модулем ML-ранжирования**": "**2.4. Интеграция Reqcore с модулем ML-ранжирования**",
    "**Разработка программного кода подготовки данных и ML-пайплайна**": "**3.1. Разработка программного кода подготовки данных и ML-пайплайна**",
    "**Реализация и дообучение (fine-tuning) модели сопоставления «вакансия-резюме»**": "**3.2. Реализация и дообучение (fine-tuning) модели сопоставления «вакансия-резюме»**",
    "**Проведение экспериментальной оценки качества алгоритма ранжирования по метрикам NDCG, Precision@K и др.**": "**4.1. Проведение экспериментальной оценки качества алгоритма ранжирования по метрикам NDCG, Precision@K и др.**",
    "**Выполнение интеграции созданного модуля в целевую ATS-систему (или её тестовый контур)**": "**4.2. Выполнение интеграции созданного модуля в целевую ATS-систему (или её тестовый контур)**",
    "**Экономическая оценка LLM scoring и bi-encoder scoring**": "**4.3. Экономическая оценка LLM scoring и bi-encoder scoring**",
    "**Офлайн-валидация bi-encoder-scoring в Reqcore**": "**4.4. Офлайн-валидация bi-encoder-scoring в Reqcore**",
}

TABLE_CAPTIONS = {
    "| Маршрут | Метод | Входная схема |": "Таблица 2.3 – Спецификация API-маршрутов FastAPI-сервиса ранжирования",
    "| Параметр | Значение в реализации | Источник |": "Таблица 3.2 – Параметры сохранённой модели cointegrated/rubert-tiny2",
    "| Параметр запуска | Значение | Источник |": "Таблица 3.3 – Гиперпараметры и фактические показатели обучения",
    "| Метрика на полном `test.tsv` | Значение | Область усреднения |": "Таблица 3.4 – Ranking-метрики на полном test split",
    "| Модель | Формула | Estimate cost / application |": "Таблица 4.3 – Сравнительная оценка стоимости LLM scoring",
    "| Уровень оценки | Что измеряется | Артефакты |": "Таблица 4.4 – Уровни офлайн-валидации bi-encoder scoring",
}


def ensure_blank_line_after_headings(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    for index, line in enumerate(lines):
        output.append(line)
        stripped = line.strip()
        if stripped.startswith("**") and stripped.endswith("**") and index + 1 < len(lines):
            next_line = lines[index + 1].strip()
            if next_line and not next_line.startswith("**") and not next_line.startswith("|") and not next_line.startswith("```"):
                output.append("")
    return "\n".join(output)


def insert_table_captions(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    for line in lines:
        inserted = False
        for marker, caption in TABLE_CAPTIONS.items():
            if line.strip().startswith(marker):
                if not output or output[-1].strip() != caption:
                    if output and output[-1].strip():
                        output.append("")
                    output.append(caption)
                    output.append("")
                inserted = True
                break
        output.append(line)
    return "\n".join(output)


def normalize_citations(text: str) -> str:
    return text.replace("[2, стр. 1]", "[2]")


def main() -> None:
    text = THESIS.read_text(encoding="utf-8")
    for old, new in HEADING_MAP.items():
        text = text.replace(old, new)
    text = normalize_citations(text)
    text = insert_table_captions(text)
    text = ensure_blank_line_after_headings(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    THESIS.write_text(text, encoding="utf-8")
    print(f"Updated {THESIS}")


if __name__ == "__main__":
    main()
