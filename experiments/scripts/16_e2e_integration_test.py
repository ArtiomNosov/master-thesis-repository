import os
import sys

# Append path to import the server
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
default_model_path = os.path.join(repo_root, "experiments", "models", "bi_encoder_rubert_tiny2")
os.environ.setdefault("MODEL_PATH", default_model_path)

# Fix stdout for Cyrillic on Windows
sys.stdout.reconfigure(encoding='utf-8')

try:
    from fastapi.testclient import TestClient
    # Import the FastAPI 'app' and internal state from the server module
    import importlib
    api_module = importlib.import_module("11_api_server")
    app = api_module.app
except ImportError as e:
    print(f"WARNING: TestClient or App import failed: {e}")
    sys.exit(1)

def run_integration_test():
    print("=====================================================")
    print("      END-TO-END INTEGRATION TEST (Stage 7.5)       ")
    print("=====================================================")

    # Initialize TestClient
    client = TestClient(app)
    
    # 1. Booting ATS FastAPI Server
    print("\n[1] Starting ATS Backend TestClient Contour")
    
    # Ensure startup event fires in TestClient to load the model
    with client:
        try:
            # 2. Ingest Candidates
            print("\n[2] Firing POST to /index/candidate (Simulating ATS Ingestion)")
            candidates = [
                {"id": "cand_1", "model_input": "Role: Frontend Web | Seniority: middle | Skills: react, typescript"},
                {"id": "cand_2", "model_input": "Role: Backend Golang | Seniority: senior | Skills: go, docker, k8s"},
                {"id": "cand_3", "model_input": "Role: Data Analytics | Seniority: junior | Skills: sql, excel, python"}
            ]
            
            for cand in candidates:
                r = client.post("/index/candidate", json=cand)
                assert r.status_code == 200, f"Failed ingestion: {r.text}"
                print(f"  -> Indexed {cand['id']} | Status: {r.status_code} | Pool Size: {r.json().get('total_pool_size')}")
    
            # 3. Search Request Pipeline
            print("\n[3] Firing POST to /search/rank (Simulating Recruiter Search Query)")
            search_query = {
                "vacancy_text": "Role: Senior Backend Dev | Skills: docker, k8s",
                "top_k": 2
            }
            
            r_search = client.post("/search/rank", json=search_query)
            
            # 4. Result Validation
            print("\n[4] Validating HTTP Response Contracts & Telemetry")
            assert r_search.status_code == 200, f"Search Rank Endpoint FAILED! {r_search.text}"
            
            res_data = r_search.json()
            assert "results" in res_data
            
            top_ans = res_data["results"][0]
            assert top_ans["candidate_id"] == "cand_2" # Must be the k8s GO dev
            
            latency_val = r_search.headers.get('X-Processing-Time-Ms', 'N/A')
            print(f"  -> Network success! Top matched candidate: {top_ans['candidate_id']} (Score: {top_ans['match_score']})")
            print(f"  -> Telemetry Headers Captured: Latency={latency_val}ms | Model={r_search.headers.get('X-Model-Version', 'Unknown')}")

            print("\n[5] Firing POST to /score (Reqcore biencoder single-pair scoring)")
            r_score = client.post("/score", json={
                "vacancy_text": "Role: Senior Backend Dev | Skills: docker, k8s",
                "candidate_text": "Role: Backend Golang | Seniority: senior | Skills: go, docker, k8s",
            })
            assert r_score.status_code == 200, f"Single score endpoint FAILED! {r_score.text}"
            score_payload = r_score.json()
            assert score_payload["status"] == "ok"
            assert isinstance(score_payload["match_score"], float)
            assert "X-Model-Version" in r_score.headers
            print(f"  -> /score success! match_score={score_payload['match_score']} | Model={r_score.headers.get('X-Model-Version')}")

            print("\n[SYSTEM CHECK: PASS] The ATS microservice correctly processed End-to-End JSON pipelines inside the API Contour.")
    
        except Exception as e:
            print(f"\n[SYSTEM CHECK: FAIL] Integration crash: {str(e)}")
            
        print("\n[6] Terminating Evaluation TestClient Contour")
        print("Integration Harness complete.")


if __name__ == "__main__":
    run_integration_test()
