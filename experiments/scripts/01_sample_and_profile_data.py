import argparse
import csv
import html
import os
import random
import re
from pathlib import Path

csv.field_size_limit(2147483647)

DEFAULT_SAMPLE_SIZE = 10000
RANDOM_SEED = 42

NEGATIVE_RE = re.compile(r"(отклон|отказ)", re.IGNORECASE)
POSITIVE_RE = re.compile(r"(на\s+рассмотр|рассмотрено|приглаш|собесед|принят)", re.IGNORECASE)


def repo_root():
    return Path(__file__).resolve().parents[2]


def normalize_event_text(text):
    return html.unescape((text or "").strip())


def classify_event(row):
    text = normalize_event_text(row.get("text"))
    if not text:
        return None

    has_negative = bool(NEGATIVE_RE.search(text))
    has_positive = bool(POSITIVE_RE.search(text))

    if has_negative and not has_positive:
        return 0
    if has_positive and not has_negative:
        return 1
    return None


def event_date(row):
    try:
        return int(row.get("date_event") or 0)
    except ValueError:
        return 0


def collect_labeled_events(events_path, sample_size):
    print(f"Collecting labeled HR events from: {events_path}")
    selected_by_pair = {}
    total_rows = 0
    explicit_rows = 0
    conflicts = 0

    with open(events_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])

        for row in reader:
            total_rows += 1
            resume_id = (row.get("id_resume") or "").strip()
            vacancy_id = (row.get("id_vac") or "").strip()
            if not resume_id or not vacancy_id or vacancy_id == "0":
                continue

            label = classify_event(row)
            if label is None:
                continue

            explicit_rows += 1
            pair_key = (resume_id, vacancy_id)
            row_date = event_date(row)
            existing = selected_by_pair.get(pair_key)

            if existing is not None and existing["label"] != label:
                conflicts += 1

            if existing is None or row_date >= existing["date"]:
                selected_by_pair[pair_key] = {
                    "row": row,
                    "label": label,
                    "date": row_date,
                }

    positives = [v for v in selected_by_pair.values() if v["label"] == 1]
    negatives = [v for v in selected_by_pair.values() if v["label"] == 0]
    random.Random(RANDOM_SEED).shuffle(positives)
    random.Random(RANDOM_SEED).shuffle(negatives)

    per_class = sample_size // 2
    selected = positives[:per_class] + negatives[:per_class]

    if len(selected) < sample_size:
        used = {((item["row"].get("id_resume") or ""), (item["row"].get("id_vac") or "")) for item in selected}
        remaining = [
            item for item in positives[per_class:] + negatives[per_class:]
            if ((item["row"].get("id_resume") or ""), (item["row"].get("id_vac") or "")) not in used
        ]
        selected.extend(remaining[:sample_size - len(selected)])

    random.Random(RANDOM_SEED).shuffle(selected)

    print(f"Scanned raw HR events: {total_rows}")
    print(f"Explicitly labeled event rows: {explicit_rows}")
    print(f"Unique explicitly labeled pairs: {len(selected_by_pair)}")
    print(f"Conflicting labels resolved by latest date_event: {conflicts}")
    print(f"Selected pairs: {len(selected)} (positive={sum(i['label'] == 1 for i in selected)}, negative={sum(i['label'] == 0 for i in selected)})")

    if not positives or not negatives:
        raise RuntimeError("Conservative label mapping did not find both positive and negative classes.")
    if not selected:
        raise RuntimeError("No labeled event pairs selected.")

    return selected, fieldnames


def extract_rows_by_id(input_path, output_path, id_column, target_ids):
    print(f"Extracting {len(target_ids)} matched rows from: {input_path}")
    found_ids = set()
    rows_written = 0

    with open(input_path, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        if not reader.fieldnames:
            raise RuntimeError(f"Missing header in {input_path}")

        with open(output_path, "w", encoding="utf-8-sig", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()

            for row in reader:
                row_id = (row.get(id_column) or "").strip()
                if row_id in target_ids:
                    writer.writerow(row)
                    found_ids.add(row_id)
                    rows_written += 1
                    if len(found_ids) == len(target_ids):
                        break

    missing = set(target_ids) - found_ids
    print(f"Saved {rows_written} rows to {output_path}; missing IDs: {len(missing)}")
    return found_ids


def write_events_sample(events, fieldnames, output_path, found_resume_ids=None, found_vacancy_ids=None):
    fieldnames = list(fieldnames)
    if "label" not in fieldnames:
        fieldnames.append("label")

    kept = 0
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for item in events:
            row = dict(item["row"])
            resume_id = (row.get("id_resume") or "").strip()
            vacancy_id = (row.get("id_vac") or "").strip()
            if found_resume_ids is not None and resume_id not in found_resume_ids:
                continue
            if found_vacancy_ids is not None and vacancy_id not in found_vacancy_ids:
                continue
            row["label"] = str(item["label"])
            writer.writerow(row)
            kept += 1

    print(f"Saved {kept} matched labeled events to {output_path}")
    if kept == 0:
        raise RuntimeError("No labeled events remain after matching resumes and vacancies.")
    return kept


def main():
    parser = argparse.ArgumentParser(description="Build event-driven matched samples from raw SuperJob TSV files.")
    root = repo_root()
    parser.add_argument("--data_dir", type=Path, default=root / "data" / "super_job")
    parser.add_argument("--sample_dir", type=Path, default=root / "data" / "sample")
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args()

    args.sample_dir.mkdir(parents=True, exist_ok=True)

    events_path = args.data_dir / "dwh_dwh_hr_received_resume_event_ods.tsv"
    resumes_path = args.data_dir / "dwh_dwh_region_resume1_ods.tsv"
    vacancies_path = args.data_dir / "dwh.hr_vacancy_ods_and_ods_hr_storage_vacancy.tsv"

    selected_events, event_fieldnames = collect_labeled_events(events_path, args.sample_size)
    resume_ids = {(item["row"].get("id_resume") or "").strip() for item in selected_events}
    vacancy_ids = {(item["row"].get("id_vac") or "").strip() for item in selected_events}

    found_resume_ids = extract_rows_by_id(
        resumes_path,
        args.sample_dir / "resumes_sample.tsv",
        "id",
        resume_ids,
    )
    found_vacancy_ids = extract_rows_by_id(
        vacancies_path,
        args.sample_dir / "vacancies_sample.tsv",
        "id",
        vacancy_ids,
    )
    write_events_sample(
        selected_events,
        event_fieldnames,
        args.sample_dir / "events_sample.tsv",
        found_resume_ids=found_resume_ids,
        found_vacancy_ids=found_vacancy_ids,
    )


if __name__ == "__main__":
    main()
