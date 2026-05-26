import os
import sys
import numpy as np

# Fix stdout for Cyrillic
sys.stdout.reconfigure(encoding='utf-8')

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("WARNING: sklearn or sentence-transformers missing.")
    sys.exit(1)

def eval_tfidf(vacancy_text: str, candidate_texts: list[str]) -> list[float]:
    # Lexical baseline
    vectorizer = TfidfVectorizer()
    all_texts = [vacancy_text] + candidate_texts
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        vac_vec = tfidf_matrix[0:1]
        cand_vecs = tfidf_matrix[1:]
        scores = cosine_similarity(vac_vec, cand_vecs)[0]
        return scores.tolist()
    except ValueError:
        return [0.0] * len(candidate_texts)

def eval_dense(model_path: str, vacancy_text: str, candidate_texts: list[str]) -> list[float]:
    model = SentenceTransformer(model_path)
    vac_emb = model.encode(vacancy_text, convert_to_tensor=True)
    cand_embs = model.encode(candidate_texts, convert_to_tensor=True)
    scores = util.cos_sim(vac_emb, cand_embs)[0]
    return scores.tolist()

if __name__ == "__main__":
    print("=== BASELINE COMPARISON EXPERIMENT (Stage 7.1) ===")
    
    # 1. Defining Mathematical Edge-Cases for evaluation
    # This demonstrates lexical mismatch fragility vs semantics
    vacancy = "Требуется сильный Python разработчик для создания REST API. Опыт работы 5 лет."
    
    # The candidates array is crafted to test algorithms
    candidates = [
        "Опытный програмист Питонист. Пишу REST веб-сервисы уже пять лет.", # Target (Rel=1) -> High semantic match, low lexical match
        "Требуется разработчик для покраски машин, требуется опыт работы.",  # Garbage (Rel=0) -> High lexical match, low semantic match
        "Начинающий Junior Python.", # Low level (Rel=0)
        "Уборщик помещений" # Noise
    ]
    
    print("\n--- 1. Baseline: TF-IDF (Lexical Exact Match) ---")
    tfidf_scores = eval_tfidf(vacancy, candidates)
    for i, score in enumerate(tfidf_scores):
        print(f"Cand {i+1}: {score:.4f} -> {candidates[i]}")
        
    print("\n--- 2. Strong Baseline: Base rubert-tiny2 (Zero-Shot) ---")
    base_scores = eval_dense("cointegrated/rubert-tiny2", vacancy, candidates)
    for i, score in enumerate(base_scores):
        print(f"Cand {i+1}: {score:.4f} -> {candidates[i]}")
        
    print("\n--- 3. Developed Model: Fine-Tuned rubert-tiny2 ---")
    finetuned_path = r"z:\repositories\master-thesis-repository\experiments\models\bi_encoder_rubert_tiny2"
    if os.path.exists(finetuned_path):
        fts_scores = eval_dense(finetuned_path, vacancy, candidates)
        for i, score in enumerate(fts_scores):
            print(f"Cand {i+1}: {score:.4f} -> {candidates[i]}")
    else:
        print(f"Warning: Model not found at {finetuned_path}")
