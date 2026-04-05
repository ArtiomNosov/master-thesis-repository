import pandas as pd
import os
import gc

def extract_sample(input_path, output_path, sample_size=10000, chunksize=1000, sep='\t'):
    print(f"Extracting sample from: {input_path}")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    reader = pd.read_csv(input_path, sep=sep, chunksize=chunksize, 
                         on_bad_lines='skip', low_memory=False, 
                         engine='c', lineterminator='\n')
    
    rows_written = 0
    for i, chunk in enumerate(reader):
        remaining = sample_size - rows_written
        if remaining <= 0:
            break
            
        to_write = chunk.head(remaining)
        mode = 'w' if i == 0 else 'a'
        header = True if i == 0 else False
        to_write.to_csv(output_path, sep='\t', index=False, mode=mode, header=header)
        
        rows_written += len(to_write)
        del chunk
        gc.collect()

    print(f"Successfully extracted {rows_written} rows to {output_path}")
    print("-" * 40)

def main():
    data_dir = r"z:\repositories\master-thesis-repository\data\super_job"
    sample_dir = r"z:\repositories\master-thesis-repository\data\sample"

    os.makedirs(sample_dir, exist_ok=True)

    files_to_process = [
        ("dwh_dwh_region_resume1_ods.tsv", "resumes_sample.tsv"),
        ("dwh.hr_vacancy_ods_and_ods_hr_storage_vacancy.tsv", "vacancies_sample.tsv"),
        ("dwh_dwh_hr_received_resume_event_ods.tsv", "events_sample.tsv")
    ]

    for in_name, out_name in files_to_process:
        in_path = os.path.join(data_dir, in_name)
        out_path = os.path.join(sample_dir, out_name)
        extract_sample(in_path, out_path, sample_size=10000, chunksize=5000)

if __name__ == "__main__":
    main()
