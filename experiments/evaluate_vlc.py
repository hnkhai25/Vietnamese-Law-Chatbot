import ujson as json
import numpy as np
from tqdm import tqdm
import os
from app.core.embedding import Embedder
from app.core.index_faiss import FaissIndex
from app.core.bm25_es import ESClient
from app.core.rerank import ReRanker
from app.settings import config

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def recall_at_k(relevant_doc_ids, retrieved_doc_ids, k):
    for doc_id in retrieved_doc_ids[:k]:
        if doc_id in relevant_doc_ids:
            return 1.0
    return 0.0
def reciprocal_rank(relevant_doc_ids, retrieved_doc_ids):
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


embedder = Embedder(config.EMBEDDING_MODEL, config.USE_GPU, normalize=True)
faiss_index = FaissIndex(config.FAISS_DIR)
es = ESClient(config.ES_HOST, config.ES_INDEX, config.ES_USER, config.ES_PASS)
reranker = ReRanker(config.RERANK_MODEL, config.USE_GPU)
try:
    faiss_index.load()
    print("FAISS index loaded.")
except Exception as e:
    print(f"Không thể load FAISS index: {e}")

W_SEM = config.HYBRID_W_SEM
W_BM25 = config.HYBRID_W_BM25


def search_semantic(query, k):
    q_emb = embedder.encode_queries([query])
    scores, idxs = faiss_index.index.search(q_emb, k)
    scores, idxs = scores[0], idxs[0]
    results = []
    for s, idx in zip(scores, idxs):
        if idx == -1:
            continue
        doc_id = faiss_index.id_map[idx]
        results.append((doc_id, float(s)))
    return results

def search_bm25(query, k):
    return [(d["doc_id"], d["score"]) for d in es.search(query, k=k)]

def search_hybrid(query, k):
    sem = search_semantic(query, max(k, 20))
    bm = search_bm25(query, max(k, 20))
    bm_map = {d[0]: d[1] for d in bm}
    hybrid = []
    for doc_id, s_sem in sem:
        s_bm = bm_map.get(doc_id, 0.0)
        final = W_SEM * s_sem + W_BM25 * s_bm
        hybrid.append((doc_id, final))
    hybrid.sort(key=lambda x: x[1], reverse=True)
    return hybrid[:k]

def search_hybrid_rerank(query, k):
    hybrid = search_hybrid(query, k)
    docs = [{"doc_id": d[0], "text": faiss_index.meta_map[d[0]]["text"]} for d in hybrid]
    reranked = reranker.rerank(query, docs)
    return [(r.get("doc_id"), r.get("score", 1.0)) for r in reranked[:k]]

def evaluate(queries_path: str, k_values=[1, 3, 5, 10]):
    queries = list(read_jsonl(queries_path))
    print(f"Loaded {len(queries)} queries from {queries_path}")

    modes = ["semantic", "bm25", "hybrid", "hybrid_rerank"]
    results = {m: {"recall": [], "mrr": []} for m in modes}

    for sample in tqdm(queries, desc="Evaluating"):
        q = sample["query"]
        relevant_doc_ids = set(sample.get("answers", []))

        if not q or not relevant_doc_ids:
            continue

        for mode in modes:
            if mode == "semantic":
                retrieved = search_semantic(q, 10)
            elif mode == "bm25":
                retrieved = search_bm25(q, 10)
            elif mode == "hybrid":
                retrieved = search_hybrid(q, 10)
            else:
                retrieved = search_hybrid_rerank(q, 10)

            retrieved_ids = [r[0] for r in retrieved]
            recall_vals = [recall_at_k(relevant_doc_ids, retrieved_ids, k) for k in k_values]
            mrr_val = reciprocal_rank(relevant_doc_ids, retrieved_ids)

            results[mode]["recall"].append(recall_vals)
            results[mode]["mrr"].append(mrr_val)

    print("\n Evaluation Results:")
    for mode in modes:
        recall_arr = np.array(results[mode]["recall"])
        if len(recall_arr) == 0:
            print(f"\n {mode.upper()}  (no valid samples)")
            continue
        recall_mean = recall_arr.mean(axis=0)
        mrr_mean = np.mean(results[mode]["mrr"])

        print(f"\n {mode.upper()} ")
        for i, k in enumerate(k_values):
            print(f"Recall@{k}: {recall_mean[i]:.3f}")
        print(f"MRR: {mrr_mean:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval models.")
    parser.add_argument("--queries", type=str, default="data\queries.jsonl" , help="Path to queries_test.jsonl")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10], help="List of k values")
    args = parser.parse_args()

    evaluate(args.queries, args.k)
