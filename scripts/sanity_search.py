import requests, os, json

API = os.getenv("API", "http://localhost:8000")

def run(q):
    for mode in ["semantic", "hybrid", "hybrid_rerank"]:
        r = requests.post(f"{API}/search", json={"query": q, "k": 5, "mode": mode}, timeout=60)
        r.raise_for_status()
        print("\n=== MODE:", mode, "===")
        for i, it in enumerate(r.json()["results"], 1):
            print(i, f"[{it.get('doc_id')}] {it['text'][:100]}...")

if __name__ == "__main__":
    run("RAG là gì?")
