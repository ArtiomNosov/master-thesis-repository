import csv
import re
import os
from pathlib import Path

csv.field_size_limit(2147483647)


def repo_root():
    return Path(__file__).resolve().parents[2]


def repair_mojibake(text):
    if not isinstance(text, str) or not text:
        return text
    if any(marker in text for marker in ("Ð", "Ñ", "ð", "Рљ", "Рџ")):
        for source_encoding in ("latin1", "cp1251"):
            try:
                candidate = text.encode(source_encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if candidate.count("Ð") + candidate.count("Ñ") + candidate.count("ð") < text.count("Ð") + text.count("Ñ") + text.count("ð"):
                return candidate
    return text

def normalize_text(text):
    if not text:
        return text

    text = repair_mojibake(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove boilerplate words
    boilerplates = [
        r"обязанности[:\s]*",
        r"требования[:\s]*",
        r"условия[:\s]*",
        r"описание вакансии[:\s]*",
        r"мы предлагаем[:\s]*",
        r"что нужно делать[:\s]*",
        r"чем предстоит заниматься[:\s]*",
        r"ожидаем от вас[:\s]*"
    ]
    for bp in boilerplates:
        text = re.sub(bp, " ", text)
        
    # Remove punctuation except for + and # (for C++, C#) and dots (versioning)
    text = re.sub(r'[^\w\s\+#\-\.]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_file(input_path, output_path, text_cols):
    print(f"Normalizing {input_path}...")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    processed = 0
    with open(input_path, 'r', encoding='utf-8-sig') as fin, open(output_path, 'w', encoding='utf-8-sig', newline='') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter='\t')
        writer.writeheader()
        
        for row in reader:
            for col in text_cols:
                if col in row and row[col]:
                    row[col] = normalize_text(row[col])
            writer.writerow(row)
            processed += 1
            
    print(f"Normalized {processed} rows. Saved to {output_path}")

if __name__ == "__main__":
    root = repo_root()
    clean_dir = root / "data" / "clean"
    preprocessed_dir = root / "data" / "preprocessed"
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    resumes_in = os.path.join(clean_dir, "resumes_cleaned.tsv")
    resumes_out = os.path.join(preprocessed_dir, "resumes_normalized.tsv")
    process_file(resumes_in, resumes_out, text_cols=["desired_profession", "best", "dop", "computer"])
    
    vacs_in = os.path.join(clean_dir, "vacancies_cleaned.tsv")
    vacs_out = os.path.join(preprocessed_dir, "vacancies_normalized.tsv")
    process_file(vacs_in, vacs_out, text_cols=["name", "profession", "candidat", "company"])
