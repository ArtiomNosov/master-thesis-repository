import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2147483647)


def repo_root():
    return Path(__file__).resolve().parents[2]


def classify_event(row):
    if row.get("label") in ("0", "1"):
        return int(row["label"])
    text = (row.get("text") or "").lower()
    has_negative = "отклон" in text or "отказ" in text
    has_positive = any(token in text for token in ("на рассмотр", "рассмотрено", "приглаш", "собесед", "принят"))
    if has_negative and not has_positive:
        return 0
    if has_positive and not has_negative:
        return 1
    return None


def stratified_split(dataset):
    grouped = defaultdict(list)
    for item in dataset:
        grouped[item[2]].append(item)

    rng = random.Random(42)
    splits = {"train.tsv": [], "val.tsv": [], "test.tsv": []}
    for label, items in grouped.items():
        rng.shuffle(items)
        n = len(items)
        if n >= 10:
            test_n = max(1, int(n * 0.1))
            val_n = max(1, int(n * 0.1))
        elif n >= 3:
            test_n = 1
            val_n = 1
        else:
            test_n = 0
            val_n = 0
        train_n = max(0, n - val_n - test_n)
        splits["train.tsv"].extend(items[:train_n])
        splits["val.tsv"].extend(items[train_n:train_n + val_n])
        splits["test.tsv"].extend(items[train_n + val_n:])

    for pairs in splits.values():
        rng.shuffle(pairs)
    return splits

def build_splits():
    root = repo_root()
    clean_dir = root / "data" / "clean"
    sample_dir = root / "data" / "sample"
    events_path = sample_dir / "events_sample.tsv"
    resumes_path = clean_dir / "resumes_cleaned.tsv"
    vacs_path = clean_dir / "vacancies_cleaned.tsv"

    out_dir = root / "data" / "splits"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading valid IDs...")
    valid_resumes = []
    if os.path.exists(resumes_path):
        with open(resumes_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if 'id' in row:
                    valid_resumes.append(row['id'])

    valid_vacs = []
    if os.path.exists(vacs_path):
        with open(vacs_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if 'id' in row:
                    valid_vacs.append(row['id'])
                    
    print(f"Loaded {len(valid_resumes)} resumes and {len(valid_vacs)} vacancies.")
    
    labeled_pairs = []
    if os.path.exists(events_path):
        print("Loading HR-labeled pairs from events...")
        with open(events_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                r_id = row.get('id_resume')
                v_id = row.get('id_vac')
                if r_id and v_id and v_id != "0":
                    label = classify_event(row)
                    if label is None:
                        continue
                    labeled_pairs.append((r_id, v_id, label))
    
    print(f"Found {len(labeled_pairs)} raw HR interactions.")
    
    # Filter to only interactions where we have both texts in our downloaded pools.
    # Note: On a 10k random sample, this intersection might be 0.
    # On the full ATS database merge, this will yield the true labeled dataset.
    dataset = []
    r_set = set(valid_resumes)
    v_set = set(valid_vacs)
    
    for r_id, v_id, label in labeled_pairs:
        if r_id in r_set and v_id in v_set:
            dataset.append((r_id, v_id, label))
            
    if len(dataset) == 0:
        raise RuntimeError("No real HR-labeled pairs intersect with cleaned resumes and vacancies. Refusing to create dummy train/val/test splits.")

    label_counts = Counter(label for _, _, label in dataset)
    print(f"Matched labeled pairs: {len(dataset)} | labels={dict(label_counts)}")
    if len(label_counts) < 2:
        raise RuntimeError(f"Need both labels for training; got label distribution {dict(label_counts)}")

    splits = stratified_split(dataset)
    
    print("Writing splits...")
    for split_name, pairs in splits.items():
        out_path = os.path.join(out_dir, split_name)
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["resume_id", "vacancy_id", "label"])
            writer.writerows(pairs)
        print(f"Wrote {len(pairs)} pairs to {split_name}")

    print("Successfully built dataset splits!")

if __name__ == "__main__":
    build_splits()
