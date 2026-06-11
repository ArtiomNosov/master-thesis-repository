import argparse
import csv
import importlib
import inspect
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    DataCollatorWithPadding = None
    Trainer = None
    TrainingArguments = None


class CrossEncoderPairDataset(torch.utils.data.Dataset if torch else object):
    """Loads vacancy-resume pairs as cross-encoder sequence-classification examples."""

    def __init__(self, split_tsv_path: str, db_path: str, tokenizer, max_length: int = 512):
        self.examples = self._load_examples(split_tsv_path, db_path)
        self.encodings = tokenizer(
            [example["vacancy_text"] for example in self.examples],
            [example["resume_text"] for example in self.examples],
            truncation=True,
            max_length=max_length,
        )
        self.labels = [example["label"] for example in self.examples]

    @staticmethod
    def _load_examples(split_tsv_path: str, db_path: str) -> list[dict]:
        examples: list[dict] = []
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        with open(split_tsv_path, "r", encoding="utf-8") as f:
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
                        "vacancy_text": vacancy_row[0],
                        "resume_text": resume_row[0],
                        "label": int(float(label)),
                    }
                )

        conn.close()
        return examples

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: values[idx] for key, values in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def build_feature_index(data_dir: str) -> str:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    dataset_module = importlib.import_module("08_model_dataset")
    return dataset_module.build_sqlite_index(
        cache_dir=os.path.join(data_dir, "splits"),
        resumes_tsv=os.path.join(data_dir, "unified", "resumes_unified.tsv"),
        vacs_tsv=os.path.join(data_dir, "unified", "vacancies_unified.tsv"),
    )


def build_training_args(args) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_average_precision",
        "greater_is_better": True,
        "report_to": [],
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["save_strategy"] = "steps"
    kwargs["eval_steps"] = args.eval_steps
    kwargs["save_steps"] = args.eval_steps

    return TrainingArguments(**kwargs)


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


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    logits_list = logits.tolist() if hasattr(logits, "tolist") else logits
    labels_list = labels.tolist() if hasattr(labels, "tolist") else labels

    positive_scores: list[float] = []
    for row in logits_list:
        if len(row) == 1:
            positive_scores.append(1.0 / (1.0 + math.exp(-row[0])))
        else:
            max_logit = max(row)
            exps = [math.exp(value - max_logit) for value in row]
            positive_scores.append(exps[1] / sum(exps))

    predictions = [1 if score >= 0.5 else 0 for score in positive_scores]
    tp = sum(1 for pred, label in zip(predictions, labels_list) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels_list) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels_list) if pred == 0 and label == 1)
    correct = sum(1 for pred, label in zip(predictions, labels_list) if pred == label)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "average_precision": average_precision(labels_list, positive_scores),
        "accuracy": correct / len(labels_list) if labels_list else 0.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def train(args) -> None:
    if Trainer is None:
        raise RuntimeError("Install torch and transformers to train the cross-encoder.")

    db_path = build_feature_index(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = CrossEncoderPairDataset(
        os.path.join(args.data_dir, "splits", "train.tsv"),
        db_path,
        tokenizer,
        max_length=args.max_length,
    )
    val_dataset = CrossEncoderPairDataset(
        os.path.join(args.data_dir, "splits", "val.tsv"),
        db_path,
        tokenizer,
        max_length=args.max_length,
    )
    test_dataset = CrossEncoderPairDataset(
        os.path.join(args.data_dir, "splits", "test.tsv"),
        db_path,
        tokenizer,
        max_length=args.max_length,
    )

    training_args = build_training_args(args)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Fine-tune a rubert-tiny2 cross-encoder on vacancy-resume pairs.")
    parser.add_argument("--model_name", default="cointegrated/rubert-tiny2")
    parser.add_argument("--data_dir", default=str(root / "data"))
    parser.add_argument("--output_dir", default=str(root / "experiments" / "models" / "cross_encoder_rubert_tiny2"))
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=50)
    train(parser.parse_args())
