import csv
import os
import json
from pathlib import Path

csv.field_size_limit(2147483647)


def repo_root():
    return Path(__file__).resolve().parents[2]


def join_non_empty(parts):
    return " ".join(part.strip() for part in parts if part and part.strip())

def format_features(input_path, output_path, is_resume=True):
    print(f"Formatting {input_path} into Unified Model Format...")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    processed = 0
    with open(input_path, 'r', encoding='utf-8-sig') as fin, open(output_path, 'w', encoding='utf-8-sig', newline='') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        
        # New output format: ID + The unified prompt format
        fieldnames = ['id', 'model_input']
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for row in reader:
            if 'id' not in row:
                continue
                
            skills_raw = row.get('extracted_skills', '[]')
            try:
                skills_list = json.loads(skills_raw)
                skills_str = ", ".join(skills_list)
            except:
                skills_str = ""
                
            seniority = row.get('extracted_seniority', "")
            
            if is_resume:
                role = row.get('desired_profession', '').strip()
                text = join_non_empty([
                    row.get('best', ''),
                    row.get('dop', ''),
                    row.get('computer', '')
                ])
            else:
                prof = row.get('profession', '').strip()
                name = row.get('name', '').strip()
                role = f"{prof} {name}".strip()
                text = join_non_empty([
                    row.get('candidat', ''),
                    row.get('company', '')
                ])
            
            # Construct unified input
            model_input = f"Role: {role} | Seniority: {seniority} | Skills: {skills_str} | Text: {text}"
            
            writer.writerow({
                'id': row['id'],
                'model_input': model_input
            })
            processed += 1
            
    print(f"Formatted {processed} rows. Saved to {output_path}")

if __name__ == "__main__":
    root = repo_root()
    features_dir = root / "data" / "features"
    unified_dir = root / "data" / "unified"
    os.makedirs(unified_dir, exist_ok=True)
    
    resumes_in = os.path.join(features_dir, "resumes_features.tsv")
    resumes_out = os.path.join(unified_dir, "resumes_unified.tsv")
    format_features(resumes_in, resumes_out, is_resume=True)
    
    vacs_in = os.path.join(features_dir, "vacancies_features.tsv")
    vacs_out = os.path.join(unified_dir, "vacancies_unified.tsv")
    format_features(vacs_in, vacs_out, is_resume=False)
