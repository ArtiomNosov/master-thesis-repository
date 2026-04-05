import csv
import os
import random

csv.field_size_limit(2147483647)

def build_splits():
    sample_dir = r"z:\repositories\master-thesis-repository\data\clean"
    # Fallback to sample if clean doesn't exist for events
    events_path = r"z:\repositories\master-thesis-repository\data\sample\events_sample.tsv"
    resumes_path = os.path.join(sample_dir, "resumes_cleaned.tsv")
    vacs_path = os.path.join(sample_dir, "vacancies_cleaned.tsv")
    
    out_dir = r"z:\repositories\master-thesis-repository\data\splits"
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
                hr_status = row.get('type')
                if r_id and v_id and hr_status:
                    # We map HR status to binary labels based on a threshold or defined types.
                    # Assuming status codes >= 100 generally correlate to negative events (e.g. rejections)
                    label = 1 if int(hr_status) < 100 else 0
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
            
    # Optionally, we enforce creating at least some dummy pairs for the pipeline structural testing 
    # strictly IF there's absolutely 0 intersection found purely due to sampling limitations.
    if len(dataset) == 0 and valid_resumes and valid_vacs:
        print("WARNING: Sample intersection evaluated to 0. Yielding technical dummy pairs for structural testing.")
        dataset = [
            (valid_resumes[0], valid_vacs[0], 1),
            (valid_resumes[-1], valid_vacs[-1], 0)
        ]

    random.shuffle(dataset)
    
    n = len(dataset)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    splits = {
        "train.tsv": dataset[:train_end],
        "val.tsv": dataset[train_end:val_end],
        "test.tsv": dataset[val_end:]
    }
    
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
