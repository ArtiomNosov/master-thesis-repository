import sys
import time
import os
from typing import List, Optional, Dict, Any

try:
    import fastapi
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    print("WARNING: FastAPI / Uvicorn missing. Install via: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Dynamically import ranking logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    ATSRanker = __import__("10_ranking_module").ATSRanker
except ImportError:
    print("ERROR: Ranking module not available.")
    raise

# --- Data Schemas ---
class CandidatePayload(BaseModel):
    id: str
    model_input: str
    metadata: Optional[Dict[str, Any]] = None

class SearchPayload(BaseModel):
    vacancy_text: str
    filters: Optional[Dict[str, str]] = None
    top_k: int = 10

class ScorePayload(BaseModel):
    vacancy_text: str
    candidate_text: str

# --- App Architecture ---
app = FastAPI(title="ATS Semantic Ranking Engine API", version="1.0.0")

# Internal runtime cache (Replacing full DB connection overhead for integration simulation)
CANDIDATE_INDEX = []
RANKER_ENGINE = None
MODEL_VERSION = None

@app.on_event("startup")
async def load_model():
    global RANKER_ENGINE, MODEL_VERSION
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH environment variable is required for the bi-encoder API server.")
    if not os.path.exists(model_path):
        raise RuntimeError(f"MODEL_PATH does not exist: {model_path}")
    print(f"Loading Integration Engine Model: {model_path}")
    RANKER_ENGINE = ATSRanker(model_path)
    MODEL_VERSION = os.path.basename(os.path.normpath(model_path))

# --- Middleware Logging ---
@app.middleware("http")
async def add_telemetry_headers(request: Request, call_next):
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Compute latency
    process_time = time.time() - start_time
    latency_ms = str(int(process_time * 1000))
    
    # Append ATS Headers securely
    response.headers["X-Processing-Time-Ms"] = latency_ms
    response.headers["X-Model-Version"] = MODEL_VERSION or "unknown"
    return response

@app.post("/index/candidate")
async def index_candidate(candidate: CandidatePayload):
    """ Ingess unstructured resumes into the active ATS dense pool. """
    cand_dict = {
        "id": candidate.id,
        "model_input": candidate.model_input
    }
    if candidate.metadata:
        cand_dict.update(candidate.metadata)
        
    CANDIDATE_INDEX.append(cand_dict)
    return {"status": "indexed", "total_pool_size": len(CANDIDATE_INDEX)}

@app.post("/search/rank")
async def search_rank(query: SearchPayload):
    """ Conducts two-stage search (Hard retrieval + Dense reranking) over memory pool. """
    if not RANKER_ENGINE:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})
        
    raw_json = RANKER_ENGINE.search(
        vacancy_text=query.vacancy_text,
        candidate_pool=CANDIDATE_INDEX,
        filters=query.filters,
        top_k=query.top_k
    )
    
    # The ranker returns JSON string, we return it as native JSON response
    import json
    return JSONResponse(content=json.loads(raw_json))

@app.post("/score")
async def score_pair(query: ScorePayload):
    """Score a single vacancy/candidate pair for Reqcore's biencoder provider."""
    if not RANKER_ENGINE:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    start = time.time()
    from sentence_transformers import util

    vacancy_emb = RANKER_ENGINE.model.encode(query.vacancy_text, convert_to_tensor=True)
    candidate_emb = RANKER_ENGINE.model.encode(query.candidate_text, convert_to_tensor=True)
    match_score = float(util.cos_sim(vacancy_emb, candidate_emb)[0][0].item())

    return {
        "status": "ok",
        "match_score": round(match_score, 4),
        "latency_ms": int((time.time() - start) * 1000),
    }

if __name__ == "__main__":
    print("Launching ATS Mock Integration Framework on Port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
