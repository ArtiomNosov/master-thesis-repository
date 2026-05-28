import argparse
import csv
import importlib
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path

# Fix stdout for Cyrillic on Windows.
sys.stdout.reconfigure(encoding="utf-8")

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None


TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Simple regular word tokenization with lowercasing for BM25."""
    return TOKEN_RE.findall((text or "").lower())


class BM25Ranker:
    """Minimal BM25 implementation without extra runtime dependencies."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))

        n_docs = len(self.doc_tokens)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query: str) -> list[float]:
        if not self.doc_tokens or self.avgdl == 0:
            return [0.0] * len(self.doc_tokens)

        query_terms = tokenize(query)
        scores: list[float] = []
        for term_freq, doc_len in zip(self.term_freqs, self.doc_lengths):
            score = 0.0
            length_norm = self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
            for term in query_terms:
                freq = term_freq.get(term, 0)
                if freq == 0:
                    continue
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (freq + length_norm)
            scores.append(score)
        return scores


def eval_bm25(vacancy_text: str, resume_texts: list[str]) -> list[float]:
    return BM25Ranker(resume_texts).score(vacancy_text)


def eval_cross_encoder(model_path: str, vacancy_text: str, resume_texts: list[str], batch_size: int = 8) -> list[float]:
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        raise RuntimeError("transformers/torch are required for cross-encoder scoring")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(resume_texts), batch_size):
            batch_resumes = resume_texts[start:start + batch_size]
            encoded = tokenizer(
                [vacancy_text] * len(batch_resumes),
                batch_resumes,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
    return scores


def eval_cross_encoder_pairs(
    model_path: str,
    vacancy_texts: list[str],
    resume_texts: list[str],
    batch_size: int = 8,
) -> list[float]:
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        raise RuntimeError("transformers/torch are required for cross-encoder scoring")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(resume_texts), batch_size):
            encoded = tokenizer(
                vacancy_texts[start:start + batch_size],
                resume_texts[start:start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
    return scores


def eval_biencoder(model_path: str, vacancy_text: str, resume_texts: list[str]) -> list[float]:
    if SentenceTransformer is None or util is None:
        raise RuntimeError("sentence-transformers is required for bi-encoder scoring")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = SentenceTransformer(model_path)
    vacancy_emb = model.encode(vacancy_text, convert_to_tensor=True)
    resume_embs = model.encode(resume_texts, convert_to_tensor=True)
    return util.cos_sim(vacancy_emb, resume_embs)[0].cpu().tolist()


def eval_biencoder_pairs(model_path: str, vacancy_texts: list[str], resume_texts: list[str]) -> list[float]:
    if SentenceTransformer is None or util is None:
        raise RuntimeError("sentence-transformers is required for bi-encoder scoring")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = SentenceTransformer(model_path)
    vacancy_embs = model.encode(vacancy_texts, convert_to_tensor=True)
    resume_embs = model.encode(resume_texts, convert_to_tensor=True)
    return util.cos_sim(vacancy_embs, resume_embs).diagonal().cpu().tolist()


def print_ranking(title: str, scores: list[float], candidates: list[str]) -> None:
    print(f"\n--- {title} ---")
    for rank, idx in enumerate(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), start=1):
        print(f"{rank}. score={scores[idx]:.4f} | Cand {idx + 1}: {candidates[idx]}")


def build_feature_index(data_dir: str) -> str:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    dataset_module = importlib.import_module("08_model_dataset")
    return dataset_module.build_sqlite_index(
        cache_dir=os.path.join(data_dir, "splits"),
        resumes_tsv=os.path.join(data_dir, "unified", "resumes_unified.tsv"),
        vacs_tsv=os.path.join(data_dir, "unified", "vacancies_unified.tsv"),
    )


def load_split_examples(data_dir: str, split: str, limit: int | None = None) -> list[dict]:
    split_path = os.path.join(data_dir, "splits", f"{split}.tsv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(split_path)

    db_path = build_feature_index(data_dir)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    examples: list[dict] = []

    with open(split_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            resume_id = row.get("resume_id")
            vacancy_id = row.get("vacancy_id")
            label = row.get("label")
            if not resume_id or not vacancy_id or label is None:
                continue

            cur.execute("SELECT model_input FROM features WHERE id=?", (vacancy_id,))
            vacancy_row = cur.fetchone()
            cur.execute("SELECT model_input FROM features WHERE id=?", (resume_id,))
            resume_row = cur.fetchone()
            if not vacancy_row or not resume_row:
                continue

            examples.append(
                {
                    "vacancy_id": vacancy_id,
                    "resume_id": resume_id,
                    "label": int(float(label)),
                    "vacancy_text": vacancy_row[0],
                    "resume_text": resume_row[0],
                }
            )
            if limit and len(examples) >= limit:
                break

    conn.close()
    return examples


def score_bm25_by_vacancy(examples: list[dict]) -> list[float]:
    scores = [0.0] * len(examples)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        groups[example["vacancy_id"]].append(idx)

    for indices in groups.values():
        vacancy_text = examples[indices[0]]["vacancy_text"]
        resume_texts = [examples[idx]["resume_text"] for idx in indices]
        group_scores = eval_bm25(vacancy_text, resume_texts)
        for idx, score in zip(indices, group_scores):
            scores[idx] = score
    return scores


def average_precision(labels: list[int], scores: list[float]) -> float:
    sorted_labels = [label for _, label in sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)]
    total_relevant = sum(sorted_labels)
    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    found = 0
    for rank, label in enumerate(sorted_labels, start=1):
        if label == 1:
            found += 1
            precision_sum += found / rank
    return precision_sum / total_relevant


def ndcg_at_k(labels: list[int], scores: list[float], k: int) -> float:
    sorted_labels = [label for _, label in sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)]
    ideal_labels = sorted(labels, reverse=True)

    def dcg(values: list[int]) -> float:
        return sum((2**label - 1) / math.log2(rank + 1) for rank, label in enumerate(values[:k], start=1))

    ideal = dcg(ideal_labels)
    return dcg(sorted_labels) / ideal if ideal > 0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_metrics(examples: list[dict], scores: list[float], k_values: tuple[int, ...] = (1, 3, 5)) -> dict:
    labels = [example["label"] for example in examples]
    metrics = {
        "pair_average_precision": average_precision(labels, scores),
        "pairs": len(examples),
    }

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        groups[example["vacancy_id"]].append(idx)

    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        mrrs = []
        for indices in groups.values():
            group_labels = [labels[idx] for idx in indices]
            group_scores = [scores[idx] for idx in indices]
            total_relevant = sum(group_labels)
            if total_relevant == 0:
                continue

            sorted_labels = [
                label for _, label in sorted(zip(group_scores, group_labels), key=lambda item: item[0], reverse=True)
            ]
            top_k = sorted_labels[:k]
            precisions.append(sum(top_k) / k)
            recalls.append(sum(top_k) / total_relevant)
            mrrs.append(next((1.0 / rank for rank, label in enumerate(sorted_labels, start=1) if label == 1), 0.0))

            if len(group_labels) > 1:
                ndcgs.append(ndcg_at_k(group_labels, group_scores, k))

        metrics[f"precision@{k}"] = mean(precisions)
        metrics[f"recall@{k}"] = mean(recalls)
        metrics[f"ndcg@{k}"] = mean(ndcgs)
        metrics[f"mrr@{k}"] = mean(mrrs)

    return metrics


def evaluate_split(args) -> None:
    examples = load_split_examples(args.data_dir, args.split, limit=args.limit)
    if not examples:
        raise RuntimeError(f"No examples loaded from {args.split}.tsv")

    vacancy_texts = [example["vacancy_text"] for example in examples]
    resume_texts = [example["resume_text"] for example in examples]

    results = {
        "split": args.split,
        "models": {},
    }
    model_scores = {
        "bm25": score_bm25_by_vacancy(examples),
    }

    if args.cross_model:
        try:
            model_scores["cross_encoder"] = eval_cross_encoder_pairs(
                args.cross_model, vacancy_texts, resume_texts, batch_size=args.batch_size
            )
        except FileNotFoundError:
            print(f"Cross-encoder model not found at {args.cross_model}; skipping.")

    if args.bi_model:
        try:
            model_scores["bi_encoder"] = eval_biencoder_pairs(args.bi_model, vacancy_texts, resume_texts)
        except FileNotFoundError:
            print(f"Bi-encoder model not found at {args.bi_model}; skipping.")

    for model_name, scores in model_scores.items():
        results["models"][model_name] = compute_metrics(examples, scores)

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Compare exactly three ranking strategies: BM25, fine-tuned cross-encoder, fine-tuned bi-encoder."
    )
    parser.add_argument("--cross_model", default=str(root / "experiments" / "models" / "cross_encoder_rubert_tiny2"))
    parser.add_argument("--bi_model", default=str(root / "experiments" / "models" / "bi_encoder_rubert_tiny2"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", default=str(root / "data"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--evaluate_split", action="store_true")
    args = parser.parse_args()

    print("=== THREE-MODEL RANKING COMPARISON (BM25 / Cross-Encoder / Bi-Encoder) ===")

    if args.evaluate_split:
        evaluate_split(args)
        return

    vacancy = "Требуется сильный Python разработчик для создания REST API. Опыт работы 5 лет."
    candidates = [
        "Опытный программист-питонист. Пишу REST веб-сервисы уже пять лет.",
        "Требуется разработчик для покраски машин, требуется опыт работы.",
        "Начинающий Junior Python.",
        "Уборщик помещений.",
    ]

    bm25_scores = eval_bm25(vacancy, candidates)
    print_ranking("1. BM25 lexical ranking baseline", bm25_scores, candidates)

    try:
        cross_scores = eval_cross_encoder(args.cross_model, vacancy, candidates, batch_size=args.batch_size)
        print_ranking("2. Fine-tuned Cross-Encoder", cross_scores, candidates)
    except Exception as exc:
        print(f"\n--- 2. Fine-tuned Cross-Encoder ---")
        print(f"Model is not available for demo scoring: {exc}")
        print("Train it with experiments/scripts/18_train_cross_encoder.py before final comparison.")

    try:
        bi_scores = eval_biencoder(args.bi_model, vacancy, candidates)
        print_ranking("3. Fine-tuned Bi-Encoder", bi_scores, candidates)
    except Exception as exc:
        print(f"\n--- 3. Fine-tuned Bi-Encoder ---")
        print(f"Model is not available for demo scoring: {exc}")


if __name__ == "__main__":
    main()
