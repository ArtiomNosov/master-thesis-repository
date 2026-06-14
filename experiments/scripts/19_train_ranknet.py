import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


sys.stdout.reconfigure(encoding="utf-8")

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def stable_hash(value: str, modulo: int, salt: str = "") -> int:
    digest = hashlib.blake2b(f"{salt}:{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def add_hashed_counts(vector: np.ndarray, offset: int, tokens: list[str], feature_dim: int, salt: str) -> None:
    if not tokens:
        return
    for token in tokens:
        idx = stable_hash(token, feature_dim, salt=salt)
        vector[offset + idx] += 1.0

    segment = vector[offset:offset + feature_dim]
    norm = np.linalg.norm(segment)
    if norm > 0:
        segment /= norm


def pair_features(vacancy_text: str, resume_text: str, feature_dim: int) -> np.ndarray:
    vacancy_tokens = tokenize(vacancy_text)
    resume_tokens = tokenize(resume_text)
    vacancy_set = set(vacancy_tokens)
    resume_set = set(resume_tokens)
    overlap_tokens = list(vacancy_set & resume_set)

    extras = 5
    vector = np.zeros(feature_dim * 3 + extras, dtype=np.float32)
    add_hashed_counts(vector, 0, vacancy_tokens, feature_dim, "vacancy")
    add_hashed_counts(vector, feature_dim, resume_tokens, feature_dim, "resume")
    add_hashed_counts(vector, feature_dim * 2, overlap_tokens, feature_dim, "overlap")

    v_len = max(len(vacancy_tokens), 1)
    r_len = max(len(resume_tokens), 1)
    union = max(len(vacancy_set | resume_set), 1)
    base = feature_dim * 3
    vector[base] = min(v_len, 1000) / 1000.0
    vector[base + 1] = min(r_len, 1000) / 1000.0
    vector[base + 2] = min(v_len, r_len) / max(v_len, r_len)
    vector[base + 3] = len(overlap_tokens) / union
    vector[base + 4] = len(overlap_tokens) / max(len(vacancy_set), 1)
    return vector


def build_feature_index(data_dir: str) -> str:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    dataset_module = importlib.import_module("08_model_dataset")
    return dataset_module.build_sqlite_index(
        cache_dir=os.path.join(data_dir, "splits"),
        resumes_tsv=os.path.join(data_dir, "unified", "resumes_unified.tsv"),
        vacs_tsv=os.path.join(data_dir, "unified", "vacancies_unified.tsv"),
    )


def load_split_examples(data_dir: str, split: str, db_path: str | None = None, limit: int | None = None) -> list[dict]:
    split_path = os.path.join(data_dir, "splits", f"{split}.tsv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(split_path)

    if db_path is None:
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


def demo_examples() -> dict[str, list[dict]]:
    rows = [
        ("py", "py-1", 1, "Role: Python backend developer | Skills: Python, FastAPI, PostgreSQL, Docker",
         "Python backend engineer. FastAPI, REST API, PostgreSQL, Docker, five years of experience."),
        ("py", "py-2", 1, "Role: Python backend developer | Skills: Python, FastAPI, PostgreSQL, Docker",
         "Backend developer, Python, Django, REST services, PostgreSQL, Docker."),
        ("py", "py-3", 0, "Role: Python backend developer | Skills: Python, FastAPI, PostgreSQL, Docker",
         "Frontend developer. React, Redux, CSS, design systems."),
        ("py", "py-4", 0, "Role: Python backend developer | Skills: Python, FastAPI, PostgreSQL, Docker",
         "Office manager with recruiting and document workflow experience."),
        ("go", "go-1", 1, "Role: Go backend developer | Skills: Go, Kubernetes, microservices, PostgreSQL",
         "Go developer. Microservices, Kubernetes, PostgreSQL, Linux, production services."),
        ("go", "go-2", 1, "Role: Go backend developer | Skills: Go, Kubernetes, microservices, PostgreSQL",
         "Golang backend engineer with Docker, Kubernetes and PostgreSQL."),
        ("go", "go-3", 0, "Role: Go backend developer | Skills: Go, Kubernetes, microservices, PostgreSQL",
         "Data analyst. SQL dashboards, Excel, reporting, product metrics."),
        ("go", "go-4", 0, "Role: Go backend developer | Skills: Go, Kubernetes, microservices, PostgreSQL",
         "Graphic designer. Figma, illustration, typography."),
        ("da", "da-1", 1, "Role: Data analyst | Skills: SQL, Python, statistics, dashboards",
         "Data analyst with SQL, Python, A/B testing, statistics and dashboard automation."),
        ("da", "da-2", 0, "Role: Data analyst | Skills: SQL, Python, statistics, dashboards",
         "Backend Java developer. Spring, Kafka, PostgreSQL."),
        ("da", "da-3", 0, "Role: Data analyst | Skills: SQL, Python, statistics, dashboards",
         "Recruiter with sourcing and interview coordination experience."),
    ]
    examples = [
        {
            "vacancy_id": vacancy_id,
            "resume_id": resume_id,
            "label": label,
            "vacancy_text": vacancy_text,
            "resume_text": resume_text,
        }
        for vacancy_id, resume_id, label, vacancy_text, resume_text in rows
    ]
    return {"train": examples, "val": examples, "test": examples}


def featurize_examples(examples: list[dict], feature_dim: int) -> np.ndarray:
    return np.vstack(
        [pair_features(example["vacancy_text"], example["resume_text"], feature_dim) for example in examples]
    )


def build_preference_pairs(
    examples: list[dict],
    max_pairs_per_query: int | None,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        groups[example["vacancy_id"]].append(idx)

    pairs: list[tuple[int, int]] = []
    eligible_queries = 0
    for indices in groups.values():
        positives = [idx for idx in indices if examples[idx]["label"] == 1]
        negatives = [idx for idx in indices if examples[idx]["label"] == 0]
        if not positives or not negatives:
            continue

        eligible_queries += 1
        query_pairs = [(pos, neg) for pos in positives for neg in negatives]
        if max_pairs_per_query and len(query_pairs) > max_pairs_per_query:
            query_pairs = rng.sample(query_pairs, max_pairs_per_query)
        pairs.extend(query_pairs)

    if not pairs:
        raise RuntimeError("RankNet needs at least one vacancy with both positive and negative pairs.")

    rng.shuffle(pairs)
    stats = {
        "eligible_queries": eligible_queries,
        "preference_pairs": len(pairs),
        "queries": len(groups),
    }
    return np.asarray(pairs, dtype=np.int64), stats


class RankNetScorer:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w1 = (rng.normal(0.0, 0.02, size=(input_dim, hidden_dim))).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = (rng.normal(0.0, 0.02, size=(hidden_dim, 1))).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        scores = h1 @ self.w2 + self.b2
        return scores[:, 0], z1, h1

    def score(self, x: np.ndarray) -> np.ndarray:
        scores, _, _ = self.forward(x)
        return scores

    def state(self) -> dict[str, np.ndarray]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.w1 = state["w1"].astype(np.float32)
        self.b1 = state["b1"].astype(np.float32)
        self.w2 = state["w2"].astype(np.float32)
        self.b2 = state["b2"].astype(np.float32)


class Adam:
    def __init__(self, params: dict[str, np.ndarray], learning_rate: float):
        self.params = params
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step_num = 0
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.step_num += 1
        for name, param in self.params.items():
            grad = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.step_num)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.step_num)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def train_epoch(
    model: RankNetScorer,
    features: np.ndarray,
    pairs: np.ndarray,
    batch_size: int,
    optimizer: Adam,
    weight_decay: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    losses: list[float] = []

    for start in range(0, len(order), batch_size):
        batch_pairs = pairs[order[start:start + batch_size]]
        pos_x = features[batch_pairs[:, 0]]
        neg_x = features[batch_pairs[:, 1]]

        pos_scores, pos_z1, pos_h1 = model.forward(pos_x)
        neg_scores, neg_z1, neg_h1 = model.forward(neg_x)
        diff = pos_scores - neg_scores
        batch_loss = np.logaddexp(0.0, -diff).mean()
        losses.append(float(batch_loss))

        grad_diff = (sigmoid(diff) - 1.0)[:, None] / len(batch_pairs)
        grad_pos_scores = grad_diff
        grad_neg_scores = -grad_diff

        grad_w2 = pos_h1.T @ grad_pos_scores + neg_h1.T @ grad_neg_scores
        grad_b2 = np.asarray([grad_pos_scores.sum() + grad_neg_scores.sum()], dtype=np.float32)

        grad_pos_h1 = grad_pos_scores @ model.w2.T
        grad_neg_h1 = grad_neg_scores @ model.w2.T
        grad_pos_z1 = grad_pos_h1 * (pos_z1 > 0)
        grad_neg_z1 = grad_neg_h1 * (neg_z1 > 0)

        grad_w1 = pos_x.T @ grad_pos_z1 + neg_x.T @ grad_neg_z1
        grad_b1 = grad_pos_z1.sum(axis=0) + grad_neg_z1.sum(axis=0)

        grads = {
            "w1": grad_w1.astype(np.float32) + weight_decay * model.w1,
            "b1": grad_b1.astype(np.float32),
            "w2": grad_w2.astype(np.float32) + weight_decay * model.w2,
            "b2": grad_b2.astype(np.float32),
        }
        optimizer.step(grads)

    return sum(losses) / len(losses)


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
            precisions.append(sum(top_k) / min(k, len(sorted_labels)))
            recalls.append(sum(top_k) / total_relevant)
            mrrs.append(next((1.0 / rank for rank, label in enumerate(sorted_labels, start=1) if label == 1), 0.0))

            if len(group_labels) > 1:
                ndcgs.append(ndcg_at_k(group_labels, group_scores, k))

        metrics[f"precision@{k}"] = mean(precisions)
        metrics[f"recall@{k}"] = mean(recalls)
        metrics[f"ndcg@{k}"] = mean(ndcgs)
        metrics[f"mrr@{k}"] = mean(mrrs)

    return metrics


def save_model(model: RankNetScorer, path: str, config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **model.state(), config=json.dumps(config, ensure_ascii=False))


def load_model(model_path: str) -> tuple[RankNetScorer, dict]:
    with np.load(model_path, allow_pickle=False) as data:
        config = json.loads(str(data["config"]))
        model = RankNetScorer(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            seed=int(config.get("seed", 42)),
        )
        model.load_state({"w1": data["w1"], "b1": data["b1"], "w2": data["w2"], "b2": data["b2"]})
    return model, config


def score_pairs(model_path: str, vacancy_texts: list[str], resume_texts: list[str]) -> list[float]:
    model, config = load_model(model_path)
    feature_dim = int(config["feature_dim"])
    features = np.vstack(
        [pair_features(vacancy, resume, feature_dim) for vacancy, resume in zip(vacancy_texts, resume_texts)]
    )
    return model.score(features).astype(float).tolist()


def run(args) -> dict:
    if args.demo:
        splits = demo_examples()
    else:
        db_path = build_feature_index(args.data_dir)
        splits = {
            "train": load_split_examples(args.data_dir, "train", db_path=db_path, limit=args.limit),
            "val": load_split_examples(args.data_dir, "val", db_path=db_path, limit=args.limit),
            "test": load_split_examples(args.data_dir, "test", db_path=db_path, limit=args.limit),
        }

    train_features = featurize_examples(splits["train"], args.feature_dim)
    val_features = featurize_examples(splits["val"], args.feature_dim)
    test_features = featurize_examples(splits["test"], args.feature_dim)
    train_pairs, pair_stats = build_preference_pairs(
        splits["train"],
        max_pairs_per_query=args.max_pairs_per_query,
        seed=args.seed,
    )

    input_dim = train_features.shape[1]
    model = RankNetScorer(input_dim=input_dim, hidden_dim=args.hidden_dim, seed=args.seed)
    optimizer = Adam(model.state(), learning_rate=args.learning_rate)

    best_val_ap = -1.0
    best_state = {name: value.copy() for name, value in model.state().items()}
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model=model,
            features=train_features,
            pairs=train_pairs,
            batch_size=args.batch_size,
            optimizer=optimizer,
            weight_decay=args.weight_decay,
            seed=args.seed + epoch,
        )
        val_scores = model.score(val_features).astype(float).tolist()
        val_metrics = compute_metrics(splits["val"], val_scores)
        history.append({"epoch": epoch, "loss": loss, "val_pair_average_precision": val_metrics["pair_average_precision"]})
        if val_metrics["pair_average_precision"] > best_val_ap:
            best_val_ap = val_metrics["pair_average_precision"]
            best_state = {name: value.copy() for name, value in model.state().items()}

    model.load_state(best_state)
    test_scores = model.score(test_features).astype(float).tolist()
    test_metrics = compute_metrics(splits["test"], test_scores)

    config = {
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "input_dim": input_dim,
        "seed": args.seed,
        "model": "numpy_ranknet_hashed_text_features",
    }
    model_path = os.path.join(args.output_dir, "ranknet.npz")
    save_model(model, model_path, config)

    result = {
        "model": "RankNet",
        "implementation": "NumPy MLP scorer with RankNet pairwise logistic loss over hashed vacancy-resume text features",
        "demo": args.demo,
        "model_path": model_path,
        "data_dir": args.data_dir,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            **pair_stats,
        },
        "history": history,
        "test_metrics": test_metrics,
    }

    if args.metrics_output:
        os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train a lightweight RankNet baseline on vacancy-resume pairs.")
    parser.add_argument("--data_dir", default=str(root / "data"))
    parser.add_argument("--output_dir", default=str(root / "experiments" / "models" / "ranknet_hashed_baseline"))
    parser.add_argument("--metrics_output", default=str(root / "experiments" / "results" / "ranknet_test_metrics.json"))
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_pairs_per_query", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Run a tiny built-in smoke dataset instead of data/splits.")
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
