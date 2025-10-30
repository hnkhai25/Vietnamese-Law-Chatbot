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
            line = line.strip()
            if line:
                yield json.loads(line)

def recall_at_k(relevant_doc_ids, retrieved_doc_ids, k):
    topk = set(retrieved_doc_ids[:k])
    return 1.0 if (topk & relevant_doc_ids) else 0.0

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
    print(f"[WARN] Không thể load FAISS index: {e}")

W_SEM = config.HYBRID_W_SEM
W_BM25 = config.HYBRID_W_BM25


meta_map = getattr(faiss_index, "meta_map", {}) or {}
if not meta_map:
    print("[WARN] faiss_index.meta_map rỗng. Bạn đã chạy index chưa?")

parent_to_children = {}
for child_id, meta in meta_map.items():
    parent = meta.get("parent_id") or meta.get("root_doc")
    if parent:
        parent_to_children.setdefault(parent, set()).add(child_id)

all_child_ids = set(meta_map.keys())

def expand_relevant_ids(relevant_ids_raw):
    expanded = set()
    for rid in relevant_ids_raw:
        if rid in all_child_ids:
            expanded.add(rid)
        else:
            children = parent_to_children.get(rid)
            if children:
                expanded.update(children)
            else:
                pass
    return expanded

def search_semantic(query, k):
    if getattr(faiss_index, "index", None) is None:
        return []
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
    hits = es.search(query, k=k) or []
    return [(d.get("doc_id"), d.get("score", 0.0)) for d in hits if d.get("doc_id")]

def normalize_bm25_scores(pairs):
    if not pairs:
        return pairs
    scores = [s for _, s in pairs]
    mn, mx = min(scores), max(scores)
    if mx <= mn:
        return [(doc, 0.0) for doc, _ in pairs]
    return [(doc, (s - mn) / (mx - mn + 1e-8)) for doc, s in pairs]

def search_hybrid(query, k, normalize_bm25=True):
    sem = search_semantic(query, max(k, 20))
    bm = search_bm25(query, max(k, 20))
    if normalize_bm25:
        bm = normalize_bm25_scores(bm)

    bm_map = {d_id: s_bm for (d_id, s_bm) in bm}
    hybrid = []
    for doc_id, s_sem in sem:
        s_bm = bm_map.get(doc_id, 0.0)
        final = W_SEM * s_sem + W_BM25 * s_bm
        hybrid.append((doc_id, final))

    sem_ids = {d for d, _ in sem}
    for d_id, s_bm in bm:
        if d_id not in sem_ids:
            final = W_SEM * 0.0 + W_BM25 * s_bm
            hybrid.append((d_id, final))

    hybrid.sort(key=lambda x: x[1], reverse=True)
    return hybrid[:k]

def search_hybrid_rerank(query, k):
    hybrid = search_hybrid(query, max(k, 20))
    docs = []
    for d_id, _ in hybrid:
        meta = meta_map.get(d_id, {})
        txt = meta.get("text", "")
        if txt:
            docs.append({"doc_id": d_id, "text": txt})

    reranked = reranker.rerank(query, docs) or []
    out = []
    for r in reranked[:k]:
        out.append((r.get("doc_id"), float(r.get("score", 1.0))))
    return out

def evaluate(queries_path: str, k_values=[1, 3, 5, 10], debug=False):
    queries = list(read_jsonl(queries_path))
    print(f"Loaded {len(queries)} queries from {queries_path}")

    modes = ["semantic", "bm25", "hybrid", "hybrid_rerank"]
    results = {m: {"recall": [], "mrr": []} for m in modes}

    empty_warned = False

    for sample in tqdm(queries, desc="Evaluating"):
        q = sample.get("query", "").strip()
        # answers: list các ID (có thể là parent_id hoặc child_id)
        relevant_raw = sample.get("answers", [])
        if not q or not relevant_raw:
            continue

        relevant_ids = expand_relevant_ids(set(relevant_raw))
        if not relevant_ids and not empty_warned:
            print("[WARN] Không mở rộng được bất kỳ child_id nào từ answers. Kiểm tra queries.jsonl xem IDs là parent hay child.")
            empty_warned = True

        for mode in modes:
            if mode == "semantic":
                retrieved = search_semantic(q, 10)
            elif mode == "bm25":
                retrieved = search_bm25(q, 10)
            elif mode == "hybrid":
                retrieved = search_hybrid(q, 10)
            else:
                retrieved = search_hybrid_rerank(q, 10)

            retrieved_ids = [r[0] for r in retrieved if r and r[0]]
            if debug:
                hit = bool(set(retrieved_ids) & relevant_ids)
                print(f"[{mode}] Q='{q[:40]}...' hit={hit} relevant={list(relevant_ids)[:3]} top1={retrieved_ids[:1]}")

            recall_vals = [recall_at_k(relevant_ids, retrieved_ids, k) for k in k_values]
            mrr_val = reciprocal_rank(relevant_ids, retrieved_ids)

            results[mode]["recall"].append(recall_vals)
            results[mode]["mrr"].append(mrr_val)

    print("\n Evaluation Results ")
    for mode in modes:
        recall_arr = np.array(results[mode]["recall"])
        if recall_arr.size == 0:
            print(f"\n {mode.upper()}  (no valid samples)")
            continue
        recall_mean = recall_arr.mean(axis=0)
        mrr_mean = float(np.mean(results[mode]["mrr"]))
        print(f"\n {mode.upper()}")
        for i, k in enumerate(k_values):
            print(f"Recall@{k}: {recall_mean[i]:.3f}")
        print(f"MRR: {mrr_mean:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate retrieval models.")
    parser.add_argument("--queries", type=str, default=r"data\queries.jsonl", help="Path to queries.jsonl")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10], help="List of k values")
    parser.add_argument("--debug", action="store_true", help="Print per-query debug hits")
    args = parser.parse_args()

    evaluate(args.queries, args.k, debug=args.debug)
