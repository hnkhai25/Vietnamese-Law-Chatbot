from fastapi import FastAPI
from app.settings import settings
from app.schemas import IndexRequest, SearchRequest
from app.core.embedding import Embedder
from app.core.index_faiss import FaissIndex
from app.core.chunking import simple_chunk
from app.core.bm25_es import ESClient
from app.core.rerank import ReRanker
import uuid

app = FastAPI(title="IR System", version="1.0.0")

embedder = Embedder(settings.EMBEDDING_MODEL, settings.USE_GPU, normalize=True)
faiss_index = FaissIndex(settings.FAISS_DIR)
try:
    faiss_index.load()
except Exception:
    pass

es = ESClient(settings.ES_HOST, settings.ES_INDEX, settings.ES_USER, settings.ES_PASS)
reranker = ReRanker(settings.RERANK_MODEL, settings.USE_GPU)

W_SEM = settings.HYBRID_W_SEM
W_BM25 = settings.HYBRID_W_BM25

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/index")
def index_docs(req: IndexRequest):
    texts, ids, metas = [], [], []
    es_docs = []
    for it in req.items:
        chunks = [it.text]
        if req.chunk:
            chunks = simple_chunk(it.text, settings.CHUNK_MAX_TOKENS, settings.CHUNK_STRIDE)

        for i, ch in enumerate(chunks):
            cid = f"{it.id}::chunk:{i}"
            texts.append(ch)
            ids.append(cid)
            meta = {"parent_id": it.id, "chunk_id": i}
            if it.meta:
                meta.update(it.meta)
            metas.append({"text": ch, **meta})

            es_docs.append({"id": cid, "text": ch, "meta": meta})

    embs = embedder.encode_passages(texts)
    faiss_index.add(embs, ids, metas)
    faiss_index.save()
    es.bulk_upsert(es_docs)
    return {"indexed": len(ids)}

@app.post("/search")
def search(req: SearchRequest):
    #  semantic
    q = embedder.encode_queries([req.query])
    sem_scores, sem_idxs = faiss_index.index.search(q, max(req.k, 20))  # get more for hybrid/rerank
    sem_scores, sem_idxs = sem_scores[0], sem_idxs[0]

    sem_hits = []
    for s, idx in zip(sem_scores, sem_idxs):
        if idx == -1: continue
        doc_id = faiss_index.id_map[idx]
        meta = faiss_index.meta_map[doc_id]
        sem_hits.append({
            "doc_id": doc_id,
            "score_semantic": float(s),
            "text": meta["text"],
            "meta": {k: v for k, v in meta.items() if k != "text"}
        })

    if req.mode == "semantic":
        return {"mode": "semantic", "results": sem_hits[:req.k]}

    #  bm25
    bm_hits = es.search(req.query, k=max(req.k, 20))
    # map for hybrid
    bm_map = {d["doc_id"]: d for d in bm_hits}

    #  hybrid weighting
    hybrid = []
    seen = set()
    for h in sem_hits:
        bm = bm_map.get(h["doc_id"])
        bm_score = bm["score"] if bm else 0.0
        final = W_SEM * h["score_semantic"] + W_BM25 * bm_score
        hybrid.append({**h, "score_bm25": float(bm_score), "score_hybrid": float(final)})
        seen.add(h["doc_id"])
    # add bm-only docs (not returned by semantic)
    for b in bm_hits:
        if b["doc_id"] in seen: continue
        final = W_SEM * 0.0 + W_BM25 * b["score"]
        hybrid.append({
            "doc_id": b["doc_id"], "text": b["text"], "meta": b["meta"],
            "score_semantic": 0.0, "score_bm25": float(b["score"]), "score_hybrid": float(final)
        })

    # sort by hybrid
    hybrid.sort(key=lambda x: x["score_hybrid"], reverse=True)
    hybrid_top = hybrid[:max(req.k, 10)]

    if req.mode == "hybrid":
        return {"mode": "hybrid", "results": hybrid_top[:req.k]}

    #  re-rank on hybrid_top
    reranked = reranker.rerank(req.query, hybrid_top, top_k=req.k)
    return {"mode": "hybrid_rerank", "results": reranked}
