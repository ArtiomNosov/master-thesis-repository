import csv
import re
import os
import sys
from pathlib import Path

csv.field_size_limit(2147483647)


def repo_root():
    return Path(__file__).resolve().parents[2]

def clean_html(raw_html):
    if not isinstance(raw_html, str):
        return raw_html
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

def mask_pii(text):
    if not isinstance(text, str) or not text:
        return text
    
    # 1. Mask Phones (basic pattern for Russian numbers)
    phone_pattern = re.compile(r'(?:\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}')
    text = phone_pattern.sub("[MASKED_PHONE]", text)
    
    # 2. Mask Emails
    email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    text = email_pattern.sub("[MASKED_EMAIL]", text)
    
    return text

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

def clean_dataset(input_path, output_path, text_cols):
    print(f"Cleaning {input_path} (Zero Dependency Mode)...")
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    processed = 0

    with open(input_path, 'r', encoding='utf-8-sig') as fin, open(output_path, 'w', encoding='utf-8-sig', newline='') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter='\t')
        writer.writeheader()
        
        try:
            for row in reader:
                # Fill missing basic values
                for k, v in row.items():
                    if not v or v.strip().lower() in ("nan", "none", "null"):
                        row[k] = ""
                
                # Apply text cleaning for specified columns
                for col in text_cols:
                    if col in row and row[col]:
                        text = row[col]
                        text = repair_mojibake(text)
                        text = clean_html(text)
                        
                        # Apply PII masking heavily on names/contacts/info
                        text = mask_pii(text)
                        
                        row[col] = text.strip()
                
                writer.writerow(row)
                processed += 1
        except Exception as e:
            print(f"Error reading file around row {processed}: {e}")

    print(f"Successfully cleaned {processed} rows. Saved to {output_path}")
    print("-" * 50)

if __name__ == "__main__":
    root = repo_root()
    sample_dir = root / "data" / "sample"
    clean_dir = root / "data" / "clean"
    os.makedirs(clean_dir, exist_ok=True)
    
    # 1. Resumes
    resumes_in = os.path.join(sample_dir, "resumes_sample.tsv")
    resumes_out = os.path.join(clean_dir, "resumes_cleaned.tsv")
    clean_dataset(
        resumes_in, resumes_out, 
        text_cols=["firstname", "lastname", "middlname", "phone1", "phone2", "email1", "desired_profession", "best", "dop", "computer", "address"]
    )
    
    # 2. Vacancies
    vacs_in = os.path.join(sample_dir, "vacancies_sample.tsv")
    vacs_out = os.path.join(clean_dir, "vacancies_cleaned.tsv")
    clean_dataset(
        vacs_in, vacs_out, 
        text_cols=["name", "profession", "candidat", "company", "contact", "address"]
    )
