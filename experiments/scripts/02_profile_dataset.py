import csv
import json
import os
import sys
from collections import Counter

csv.field_size_limit(2147483647)

def profile_dataset(input_path, output_path, cols_to_profile, text_col=None):
    print(f"Profiling {input_path} (Zero Dependency Mode)...")
    
    total_rows = 0
    total_text_len = 0
    valid_text_count = 0
    
    distributions = {col: Counter() for col in cols_to_profile}

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        try:
            for row in reader:
                total_rows += 1
                
                if text_col and text_col in row and row[text_col]:
                    val = row[text_col].strip()
                    if val and val.lower() not in ("nan", "none", ""):
                        total_text_len += len(val)
                        valid_text_count += 1
                        
                for col in cols_to_profile:
                    if col in row and row[col]:
                        val = row[col].strip()
                        if val and val.lower() not in ("nan", "none", ""):
                            distributions[col][val] += 1
                            
        except Exception as e:
            print(f"Error reading file around row {total_rows}: {e}")

    avg_len = total_text_len / valid_text_count if valid_text_count > 0 else 0
    
    report = {
        "file": input_path,
        "volume": total_rows,
        "average_length": {
            "column": text_col,
            "value": round(avg_len, 2)
        },
        "distributions": {}
    }
    
    for col, counter in distributions.items():
        report["distributions"][col] = counter.most_common(20)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print(f"Saved profile report to {output_path}")
    print("-" * 50)

if __name__ == "__main__":
    sample_dir = r"z:\repositories\master-thesis-repository\data\sample"
    out_dir = r"z:\repositories\master-thesis-repository\experiments\profiling_reports"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Resumes
    resumes_in = os.path.join(sample_dir, "resumes_sample.tsv")
    resumes_out = os.path.join(out_dir, "resumes_profile.json")
    profile_dataset(
        resumes_in, resumes_out, 
        cols_to_profile=["desired_profession", "town", "type_of_work", "pol", "experience"],
        text_col="desired_profession"
    )
    
    # 2. Vacancies
    vacs_in = os.path.join(sample_dir, "vacancies_sample.tsv")
    vacs_out = os.path.join(out_dir, "vacancies_profile.json")
    profile_dataset(
        vacs_in, vacs_out, 
        cols_to_profile=["profession", "name", "town", "compensation", "experience", "type_of_work"],
        text_col="name"
    )
    
    # 3. Events
    events_in = os.path.join(sample_dir, "events_sample.tsv")
    events_out = os.path.join(out_dir, "events_profile.json")
    profile_dataset(
        events_in, events_out, 
        cols_to_profile=["type"],
        text_col=None
    )
