import argparse
import importlib
import math
import os
import sys
from pathlib import Path

# Fix stdout for Cyrillic on Windows
sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
comparison = importlib.import_module("12_baseline_comparison")

# --- Metric Implementations ---
def compute_precision_at_k(actual_labels, k=3):
    top_k = actual_labels[:k]
    return sum(top_k) / k

def compute_recall_at_k(actual_labels, total_relevant, k=3):
    top_k = actual_labels[:k]
    return sum(top_k) / total_relevant if total_relevant > 0 else 0

def compute_mrr(actual_labels):
    for rank, label in enumerate(actual_labels, start=1):
        if label == 1:
            return 1 / rank
    return 0.0

def evaluate_predictions(y_true, scores, k=3):
    """ Orchestrates IR metrics calculation. """
    # Sort pairs by descending predicted score
    sorted_pairs = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
    sorted_y_true = [y for _, y in sorted_pairs]
    
    total_rel = sum(y_true)
    
    ideal_labels = sorted(y_true, reverse=True)

    def dcg(values):
        return sum((2**label - 1) / math.log2(rank + 1) for rank, label in enumerate(values[:k], start=1))

    ideal = dcg(ideal_labels)
    ndcg = dcg(sorted_y_true) / ideal if ideal > 0 else 0.0
    
    prec_k = compute_precision_at_k(sorted_y_true, k=k)
    rec_k = compute_recall_at_k(sorted_y_true, total_rel, k=k)
    mrr = compute_mrr(sorted_y_true)
    
    return {"NDCG@K": ndcg, "Precision@K": prec_k, "Recall@K": rec_k, "MRR": mrr}

def run_evaluation(args):
    print("==================================================")
    print("    THREE-MODEL RANKING METRICS EXPERIMENT 7.3    ")
    print("==================================================")

    # Demo ranking group. The final thesis evaluation must be run on test.tsv.
    vacancy_text = "Role: Go Backend Developer | Seniority: middle | Skills: golang, postgresql, docker, microservices"

    candidates = [
        "Go-разработчик. 3 года опыта. Микросервисы, PostgreSQL.",            # True Relevant (1)
        "Разработчик на Python и немного Go. Знаю Docker.",                    # Marginal (0)
        "Go Backend Developer | Skills: golang, postgresql, microservices",    # True Relevant (1)
        "Frontend Developer. Пишу React, Redux, Docker.",                      # Irrelevant (0)
        "Опытный системный администратор, поднимаю БД, Linux."                 # Irrelevant (0)
    ]
    y_true = [1, 0, 1, 0, 0] # Boolean labels mapping
    K = 3 # Evaluate Top 3 logic

    model_scores = {
        "1. BM25 lexical ranking baseline": comparison.eval_bm25(vacancy_text, candidates),
    }

    try:
        model_scores["2. Fine-tuned Cross-Encoder"] = comparison.eval_cross_encoder(
            args.cross_model, vacancy_text, candidates, batch_size=args.batch_size
        )
    except Exception as exc:
        print(f"\n--- 2. Fine-tuned Cross-Encoder ---")
        print(f"Skipped: {exc}")
        print("Train experiments/models/cross_encoder_rubert_tiny2 with 18_train_cross_encoder.py.")

    try:
        model_scores["3. Fine-tuned Bi-Encoder"] = comparison.eval_biencoder(
            args.bi_model, vacancy_text, candidates
        )
    except Exception as exc:
        print(f"\n--- 3. Fine-tuned Bi-Encoder ---")
        print(f"Skipped: {exc}")

    for model_name, scores in model_scores.items():
        metrics = evaluate_predictions(y_true, scores, k=K)
        print(f"\n--- {model_name} ---")
        for metric, val in metrics.items():
            print(f"{metric}: {val:.4f}")

    print("\n[CONCLUSIONS OVERVIEW]")
    print("BM25 measures lexical overlap without training.")
    print("Cross-Encoder estimates the quality ceiling for expensive pairwise scoring.")
    print("Bi-Encoder measures scalable semantic retrieval with independently cached resume embeddings.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compute demo IR metrics for the three-model comparison.")
    parser.add_argument("--cross_model", default=str(root / "experiments" / "models" / "cross_encoder_rubert_tiny2"))
    parser.add_argument("--bi_model", default=str(root / "experiments" / "models" / "bi_encoder_rubert_tiny2"))
    parser.add_argument("--batch_size", type=int, default=8)
    run_evaluation(parser.parse_args())
