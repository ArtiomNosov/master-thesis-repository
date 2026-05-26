import sys
import json
import torch
import os

# Fix print encoding for Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

# Attempt native import since environment satisfies requirements
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("WARNING: sentence-transformers is missing. Ensure the virtual environment is active.")
    sys.exit(1)

class ATSRanker:
    """
    Two-Stage Application Tracking System Search Engine.
    Stage 1: Hard Filtering (Retrieval)
    Stage 2: Semantic Reranking (Bi-Encoder Dense Matching)
    """
    def __init__(self, model_path_or_name: str):
        print(f"Loading Semantic Matcher from: {model_path_or_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_path_or_name, device=self.device)
        print("Model loaded successfully.")

    def _apply_hard_filters(self, candidates: list[dict], filters: dict) -> list[dict]:
        """
        Stage 1: Discard candidates missing mandatory constraints.
        Matches filters exactly if provided.
        """
        if not filters:
            return candidates
            
        filtered_pool = []
        for cand in candidates:
            is_valid = True
            for k, required_val in filters.items():
                if cand.get(k) != required_val:
                    is_valid = False
                    break
            if is_valid:
                filtered_pool.append(cand)
                
        return filtered_pool

    def search(self, vacancy_text: str, candidate_pool: list[dict], filters: dict = None, top_k: int = 10) -> str:
        """
        Executes the full pipeline and returns JSON of top candidates.
        candidate_pool expects format: [{'id': '123', 'model_input': 'Role: ...', 'town': '14', ...}]
        """
        import time
        start = time.time()
        
        # --- STAGE 1: Retrieval ---
        # Rapid elimination based on categorical parameters (Reduces GPU requirement by 90%)
        filtered_candidates = self._apply_hard_filters(candidate_pool, filters)
        
        if not filtered_candidates:
            return json.dumps({"status": "ok", "latency_ms": 0, "results": []})
            
        # --- STAGE 2: Reranking ---
        # Extract the semantic 'model_input' sequences
        candidate_texts = [c.get('model_input', '') for c in filtered_candidates]
        
        # Compute embeddings
        # For a massive ATS, candidate_texts would be pre-embedded in FAISS.
        # Since we simulate dynamic injection, we encode on the fly.
        vacancy_emb = self.model.encode(vacancy_text, convert_to_tensor=True)
        candidate_embs = self.model.encode(candidate_texts, convert_to_tensor=True)
        
        # Calculate Cosine Similarities via PyTorch backend
        cos_scores = util.cos_sim(vacancy_emb, candidate_embs)[0]
        
        # Sort scores descending
        top_results = torch.topk(cos_scores, k=min(top_k, len(filtered_candidates)))
        
        final_list = []
        for score, idx in zip(top_results[0], top_results[1]):
            cand_data = filtered_candidates[idx]
            match_score = float(score.item())
            
            # Form final machine-readable wrapper
            final_list.append({
                "candidate_id": cand_data.get('id', 'unknown'),
                "match_score": round(match_score, 4),
                "summary": cand_data.get('model_input', '')[:150] + "..." # truncated preview
            })
            
        total_time_ms = int((time.time() - start) * 1000)
        
        output = {
            "status": "ok",
            "latency_ms": total_time_ms,
            "filters_applied": filters,
            "candidates_evaluated": len(filtered_candidates),
            "results": final_list
        }
        
        return json.dumps(output, indent=4, ensure_ascii=False)


# =========================================================
# Self-Contained Execution & Structural Integrity Test
# =========================================================
if __name__ == "__main__":
    model_path = r"z:\repositories\master-thesis-repository\experiments\models\bi_encoder_rubert_tiny2"
    
    # If model wasn't saved locally due to dummy data, fallback to base weights
    if not os.path.exists(model_path):
        print("Fine-tuned model not found locally, falling back to base model for pipeline structural test.")
        model_path = "cointegrated/rubert-tiny2"
        
    ranker = ATSRanker(model_path)
    
    dummy_vacancy = "Role: Senior Backend Developer | Seniority: linux | Skills: python, django, rest api, postgresql | Text: Требуется опытный программист для разработки высоконагруженного сервиса."
    
    # Simulate a NoSQL Document Database mapping
    dummy_pool = [
        {"id": "RES-01", "town": "Moscow", "model_input": "Role: Data Scientist | Seniority: Junior | Skills: python, pandas | Text: Начинающий специалист."},
        {"id": "RES-02", "town": "London", "model_input": "Role: Backend Developer | Seniority: Senior | Skills: python, django | Text: 5 лет опыта разработки REST API."},
        {"id": "RES-03", "town": "Moscow", "model_input": "Role: Middle Python Dev | Seniority: Middle | Skills: python, django, linux | Text: Пишу бекенд на джанго."},
        {"id": "RES-04", "town": "Moscow", "model_input": "Role: Frontend Web | Seniority: Senior | Skills: react, javascript | Text: Создаю пользовательские интерфейсы."}
    ]
    
    print("\nExecuting Test 1: No Filters (Search among all)")
    result1 = ranker.search(dummy_vacancy, dummy_pool, top_k=2)
    print(result1)
    
    print("\nExecuting Test 2: Mandatory Filter (town == Moscow)")
    result2 = ranker.search(dummy_vacancy, dummy_pool, filters={"town": "Moscow"}, top_k=2)
    print(result2)
