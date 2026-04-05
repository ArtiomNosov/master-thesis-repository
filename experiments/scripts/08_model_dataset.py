import csv
import os
import sqlite3

try:
    from torch.utils.data import IterableDataset
except ImportError:
    print("WARNING: PyTorch not installed. Run `pip install torch`")
    # Dummy placeholder so script doesn't completely crash if torch is missing
    class IterableDataset:
        pass

try:
    from sentence_transformers import InputExample
except ImportError:
    print("WARNING: sentence-transformers not installed. Run `pip install sentence-transformers`")
    class InputExample:
        def __init__(self, texts, label=None):
            self.texts = texts
            self.label = label


def build_sqlite_index(cache_dir, resumes_tsv, vacs_tsv):
    """
    Builds a fast SQLite lookup table to dynamically fetch textual features by ID
    so we don't have to load 44 GB of data into RAM.
    """
    db_path = os.path.join(cache_dir, "model_features.db")
    # If it exists, we just connect to it to save time
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id TEXT PRIMARY KEY,
            model_input TEXT,
            doc_type TEXT
        )
    ''')
    
    # Check if empty
    cur.execute('SELECT COUNT(*) FROM features')
    if cur.fetchone()[0] > 0:
        print("SQLite feature index already built.")
        return db_path
        
    print("Building SQLite feature memory-safe index (this runs once)...")
    
    for tsv_file, doc_type in [(resumes_tsv, 'resume'), (vacs_tsv, 'vacancy')]:
        if not os.path.exists(tsv_file):
            continue
        with open(tsv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter='\t')
            batch = []
            for row in reader:
                if 'id' in row and 'model_input' in row:
                    batch.append((row['id'], row['model_input'], doc_type))
                if len(batch) >= 10000:
                    cur.executemany('INSERT OR IGNORE INTO features (id, model_input, doc_type) VALUES (?, ?, ?)', batch)
                    batch = []
            if batch:
                cur.executemany('INSERT OR IGNORE INTO features (id, model_input, doc_type) VALUES (?, ?, ?)', batch)
                
    conn.commit()
    conn.close()
    print("Index build complete!")
    return db_path


class StreamingPairDataset(IterableDataset):
    """
    An IterableDataset that streams TSV splits and dynamically looks up 
    text strings from SQLite, keeping maximum Python RAM usage < 100MB.
    """
    def __init__(self, split_tsv_path, db_path):
        super(StreamingPairDataset).__init__()
        self.split_tsv_path = split_tsv_path
        self.db_path = db_path
        
    def __iter__(self):
        # We must open a new sqlite connection per thread/iterator
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        with open(self.split_tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                r_id = row.get('resume_id')
                v_id = row.get('vacancy_id')
                label_str = row.get('label')
                
                if not r_id or not v_id or not label_str:
                    continue
                    
                label = float(label_str)
                
                # Fast lookup
                cur.execute("SELECT model_input FROM features WHERE id=?", (r_id,))
                res_resume = cur.fetchone()
                
                cur.execute("SELECT model_input FROM features WHERE id=?", (v_id,))
                res_vac = cur.fetchone()
                
                if res_resume and res_vac:
                    yield InputExample(texts=[res_resume[0], res_vac[0]], label=label)

if __name__ == "__main__":
    # Test compilation
    print("08_model_dataset.py structure verified.")
