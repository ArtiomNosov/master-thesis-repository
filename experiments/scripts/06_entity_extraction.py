import csv
import re
import os
import json
from pathlib import Path

csv.field_size_limit(2147483647)


def repo_root():
    return Path(__file__).resolve().parents[2]

SENIORITY_MAP = {
    'intern/junior': [r'\b(junior|джуниор|младший|начинающий|стажер|intern)\b'],
    'middle': [r'\b(middle|мидл|миддл)\b'],
    'senior': [r'\b(senior|сеньор|синьор|старший|ведущий)\b'],
    'lead': [r'\b(lead|лид|руководитель|главный|tech lead|team lead)\b']
}

# A robust subset of generic technical and methodological skills to extract
SKILLS_LIST = [
    'python', 'sql', 'java', 'react', 'docker', 'pandas', 'c\+\+', 'c#', 'git', 'linux', 'bash', 'aws', 
    'kubernetes', 'tensorflow', 'pytorch', 'machine learning', 'data science', 'go', 'node.js', 
    'javascript', 'typescript', 'php', 'ruby', 'agile', 'scrum', 'jira', 'confluence', 'postgresql', 
    'mysql', 'mongodb', 'redis', 'devops', 'ci cd', 'ansible', 'azure', 'gcp', 'spring', 'django', 'flask', 
    'fastapi', 'vue', 'angular', 'html', 'css', 'excel', 'powerbi', 'tableau', 'erp', '1c'
]

def extract_entities(text):
    if not text:
        return '[]', ''
    
    extracted_skills = set()
    extracted_seniority = ""
    
    # 1. Extract Seniority
    for level, patterns in SENIORITY_MAP.items():
        for p in patterns:
            if re.search(p, text):
                extracted_seniority = level
                break # Just take the first match
        if extracted_seniority:
            break
            
    # 2. Extract Skills
    for skill in SKILLS_LIST:
        # Avoid partial word matches like "go" inside "good"
        pattern = r'\b' + skill + r'\b'
        if re.search(pattern, text):
            extracted_skills.add(skill.replace('\\', ''))
            
    return json.dumps(list(extracted_skills)), extracted_seniority

def process_features(input_path, output_path, extract_cols):
    print(f"Extracting entities for {input_path}...")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    processed = 0
    with open(input_path, 'r', encoding='utf-8-sig') as fin, open(output_path, 'w', encoding='utf-8-sig', newline='') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        
        # Add new columns to header
        fieldnames = reader.fieldnames + ['extracted_skills', 'extracted_seniority']
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for row in reader:
            combined_text = " ".join([row.get(col, "") for col in extract_cols if row.get(col)])
            skills, seniority = extract_entities(combined_text)
            
            row['extracted_skills'] = skills
            row['extracted_seniority'] = seniority
            
            writer.writerow(row)
            processed += 1
            
    print(f"Extraction complete for {processed} rows. Saved to {output_path}")

if __name__ == "__main__":
    root = repo_root()
    preprocessed_dir = root / "data" / "preprocessed"
    features_dir = root / "data" / "features"
    os.makedirs(features_dir, exist_ok=True)
    
    resumes_in = os.path.join(preprocessed_dir, "resumes_normalized.tsv")
    resumes_out = os.path.join(features_dir, "resumes_features.tsv")
    process_features(resumes_in, resumes_out, extract_cols=["desired_profession", "best", "dop", "computer"])
    
    vacs_in = os.path.join(preprocessed_dir, "vacancies_normalized.tsv")
    vacs_out = os.path.join(features_dir, "vacancies_features.tsv")
    process_features(vacs_in, vacs_out, extract_cols=["name", "profession", "candidat", "company"])
