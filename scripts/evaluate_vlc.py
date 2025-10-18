import ujson as json
import numpy as np
import random
import os
from tqdm import tqdm

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

def recall_at_k(relevant_doc_id, retrieved_doc_ids, k):
    """Trả về 1 nếu doc đúng nằm trong top-k"""
    return 1.0 if relevant_doc_id in retrieved_doc_ids[:k] else 0.0

def reciprocal_rank(relevant_doc_id, retrieved_doc_ids):
    """MRR = 1 / rank của doc đúng"""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == relevant_doc_id:
            return 1.0 / rank
    return 0.0


embedder = Embedder(config.EMBEDDING_MODEL, config.USE_GPU, normalize=True)
faiss_index = FaissIndex(config.FAISS_DIR)
try:
    faiss_index.load()
    print("FAISS index loaded.")
except Exception as e:
    print(f"Không thể load FAISS index: {e}")

es = ESClient(config.ES_HOST, config.ES_INDEX, config.ES_USER, config.ES_PASS)
reranker = ReRanker(config.RERANK_MODEL, config.USE_GPU)

W_SEM = config.HYBRID_W_SEM
W_BM25 = config.HYBRID_W_BM25


data = list(read_jsonl("data/corpus.jsonl"))
random.shuffle(data)
split_idx = int(len(data) * 0.7)
train_data = data[:split_idx]
val_data = data[split_idx:]
for d in val_data:
    d["query"] = d.get("meta", {}).get("query_example", "")
    
print(f"Train size: {len(train_data)} | Val size: {len(val_data)}")


def search_semantic(query, k):
    q_emb = embedder.encode_queries([query])
    scores, idxs = faiss_index.index.search(q_emb, k)
    scores, idxs = scores[0], idxs[0]
    results = []
    for s, idx in zip(scores, idxs):
        if idx == -1:
            continue
        doc_id = faiss_index.id_map[idx]
        results.append((doc_id, s))
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


modes = ["semantic", "bm25", "hybrid", "hybrid_rerank"]
k_values = [1, 3, 5, 10]

results = {m: {"recall": [], "mrr": []} for m in modes}

for sample in tqdm(val_data, desc="Evaluating"):
    q = sample["query"]
    true_doc_id = sample.get("id") or None

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
        recall_vals = [recall_at_k(true_doc_id, retrieved_ids, k) for k in k_values]
        mrr_val = reciprocal_rank(true_doc_id, retrieved_ids)

        results[mode]["recall"].append(recall_vals)
        results[mode]["mrr"].append(mrr_val)


print("\nEvaluation Results:")
for mode in modes:
    recall_arr = np.array(results[mode]["recall"])
    recall_mean = recall_arr.mean(axis=0)
    mrr_mean = np.mean(results[mode]["mrr"])

    print(f"\n--- {mode.upper()} ---")
    for i, k in enumerate(k_values):
        print(f"Recall@{k}: {recall_mean[i]:.3f}")
    print(f"MRR: {mrr_mean:.3f}")
