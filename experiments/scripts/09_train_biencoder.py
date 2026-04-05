import os
import argparse
import sys
import importlib
from datetime import datetime

try:
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.evaluation import BinaryClassificationEvaluator
except ImportError:
    print("WARNING: PyTorch/sentence-transformers missing.")
    DataLoader, SentenceTransformer, losses, BinaryClassificationEvaluator = None, None, None, None

def train(args):
    # Dynamically import the dataset module since it starts with a number
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    dataset_module = importlib.import_module("08_model_dataset")
    build_sqlite_index = dataset_module.build_sqlite_index
    StreamingPairDataset = dataset_module.StreamingPairDataset
    
    print(f"[{datetime.now()}] Initializing Model: {args.model_name}")
    
    if SentenceTransformer is None:
        print("SentenceTransformer not imported. Exiting model setup gracefully.")
        return
        
    model = SentenceTransformer(args.model_name)
    
    # 1. Build SQLite cache mapping (so we don't load 44GB RAM)
    resumes_tsvs = os.path.join(args.data_dir, "unified", "resumes_unified.tsv")
    vacs_tsvs = os.path.join(args.data_dir, "unified", "vacancies_unified.tsv")
    
    db_path = build_sqlite_index(
        cache_dir=os.path.join(args.data_dir, "splits"),
        resumes_tsv=resumes_tsvs,
        vacs_tsv=vacs_tsvs
    )
    
    # 2. Setup streaming DataLoaders
    train_tsv = os.path.join(args.data_dir, "splits", "train.tsv")
    val_tsv = os.path.join(args.data_dir, "splits", "val.tsv")
    
    train_dataset = StreamingPairDataset(train_tsv, db_path)
    # the DataLoader wraps the iterable generator automatically batching it
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size) 
    
    val_dataset = StreamingPairDataset(val_tsv, db_path)
    val_examples = list(val_dataset) # For validation, loading a small test subset into memory is acceptable
    
    # Validation Evaluator
    if len(val_examples) > 0:
        evaluator = BinaryClassificationEvaluator.from_input_examples(val_examples, name='val-binary')
    else:
        evaluator = None
    
    # 3. Loss setup
    # Contrastive Loss is optimal for Siamese (Resume, Vacancy, Distance_Label) format.
    train_loss = losses.ContrastiveLoss(model=model)
    
    out_model_path = os.path.join(args.output_dir, "bi_encoder_rubert_tiny2")
    os.makedirs(out_model_path, exist_ok=True)
    
    # 4. Train
    print(f"[{datetime.now()}] Starting Training...")
    fit_args = {
        'train_objectives': [(train_dataloader, train_loss)],
        'epochs': args.epochs,
        'warmup_steps': args.warmup_steps,
        'output_path': out_model_path,
        'save_best_model': False
    }
    
    if evaluator is not None:
        fit_args['evaluator'] = evaluator
        fit_args['evaluation_steps'] = 100
        fit_args['save_best_model'] = True
        
    model.fit(**fit_args)
    print(f"[{datetime.now()}] Training complete. Best model saved to {out_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Safe Semantic Ranking Training")
    parser.add_argument("--model_name", type=str, default="cointegrated/rubert-tiny2", help="Base HuggingFace model")
    parser.add_argument("--data_dir", type=str, default=r"z:\repositories\master-thesis-repository\data", help="Root data dir")
    parser.add_argument("--output_dir", type=str, default=r"z:\repositories\master-thesis-repository\experiments\models", help="Model saving dir")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size (Streamed)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR Warmup steps")
    
    args = parser.parse_args()
    train(args)
