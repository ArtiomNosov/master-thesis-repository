import sys
import time
import tracemalloc
import torch

# Fix stdout for Cyrillic on Windows
sys.stdout.reconfigure(encoding='utf-8')

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("WARNING: Missing sentence-transformers package")
    sys.exit(1)

def run_performance_test():
    print("=====================================================")
    print("      DEPLOYAIBILITY BENCHMARK (Stage 7.4)          ")
    print("=====================================================")

    # Setup
    vacancy_text = "Role: Senior Data Scientist | Seniority: senior | Skills: python, pytorch, nlp | Text: Seeking machine learning expert."
    base_candidate = "Role: Machine Learning Engineer | Seniority: middle | Skills: python, tensorflow, nlp | Text: I build models."

    # 1. Booting Memory Profiler 
    print("\n--- 1. Memory Overhead Profiling ---")
    tracemalloc.start()
    
    # Load Model simulating Server Startup
    t0 = time.time()
    model = SentenceTransformer("cointegrated/rubert-tiny2")
    startup_time = time.time() - t0
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Model Init Time      : {startup_time * 1000:.2f} ms")
    print(f"Peak RAM Allocation  : {peak / 10**6:.2f} MB")
    if peak / 10**6 < 1000:
        print("[DEPLOYMENT CHECK: PASS] Model successfully respects the < 1 GB hard limit.")

    # 2. Scalability Latency Stress Test
    print("\n--- 2. Production Latency Assessment ---")
    pools_to_test = [10, 100, 1000, 5000]
    
    # Pre-Encoding the Vacancy once (ATS typical behavior)
    v_enc = model.encode([vacancy_text], convert_to_tensor=True)

    for N in pools_to_test:
        simulated_pool = [base_candidate] * N
        
        # We start the stopwatch just for the processing loop (Reranking Phase)
        start_t = time.time()
        c_encs = model.encode(simulated_pool, convert_to_tensor=True, batch_size=256)
        scores = util.cos_sim(v_enc, c_encs)
        # Add indexing simulation
        sorted_indices = torch.argsort(scores, dim=-1, descending=True)
        end_t = time.time()
        
        latency = (end_t - start_t) * 1000
        print(f"Pool Size: {N:>5} cands | Rerank Latency: {latency:>8.2f} ms | Cost per element: {(latency/N):.3f} ms")
    
    # 3. Throughput QPS Benchmark
    print("\n--- 3. Throughput QPS Extrapolation ---")
    # For a standard job platform where N=100 per Vacancy (after Stage 1 Retrieval filtering)
    base_latency_100 = 0.0
    loops = 20
    for _ in range(loops):
        st = time.time()
        _c_encs = model.encode([base_candidate] * 100, convert_to_tensor=True, batch_size=100)
        _s = util.cos_sim(v_enc, _c_encs)
        base_latency_100 += (time.time() - st)
        
    avg_s = base_latency_100 / loops
    qps = 1.0 / avg_s
    print(f"Assuming Candidate Pool = 100 (Post Stage 1 Hard-Filter)")
    print(f"Average Request Time : {avg_s * 1000:.2f} ms")
    print(f"Requests Per Second (QPS) without GPU: {qps:.1f} req/s")


if __name__ == "__main__":
    run_performance_test()
