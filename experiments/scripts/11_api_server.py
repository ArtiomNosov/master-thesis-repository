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
    print("WARNING: Ranking module not available.")
    # Mock ranker
    class ATSRanker:
        def __init__(self, m): pass
        def search(self, v, cp, f, tk): return "{}"

# --- Data Schemas ---
class CandidatePayload(BaseModel):
    id: str
    model_input: str
    metadata: Optional[Dict[str, Any]] = None

class SearchPayload(BaseModel):
    vacancy_text: str
    filters: Optional[Dict[str, str]] = None
    top_k: int = 10

# --- App Architecture ---
app = FastAPI(title="ATS Semantic Ranking Engine API", version="1.0.0")

# Internal runtime cache (Replacing full DB connection overhead for integration simulation)
CANDIDATE_INDEX = []
RANKER_ENGINE = None
MODEL_VERSION = "rubert-tiny2-v1"

@app.on_event("startup")
async def load_model():
    global RANKER_ENGINE
    model_path = r"z:\repositories\master-thesis-repository\experiments\models\bi_encoder_rubert_tiny2"
    if not os.path.exists(model_path):
        model_path = "cointegrated/rubert-tiny2"
    print(f"Loading Integration Engine Model: {model_path}")
    RANKER_ENGINE = ATSRanker(model_path)

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
    response.headers["X-Model-Version"] = MODEL_VERSION
    
    # System logging for observability
    print(f"[ATS_LOGGER] Path: {request.url.path} | Time: {latency_ms}ms | Model: {MODEL_VERSION}")
    
    return response

# --- Endpoints ---
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

if __name__ == "__main__":
    print("Launching ATS Mock Integration Framework on Port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
