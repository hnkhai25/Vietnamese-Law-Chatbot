from fastapi import FastAPI
from app import config, IndexRequest, SearchRequest
from app.core import Embedder, FaissIndex, ESClient, ReRanker, hierarchical_chunk
import uuid
import google.generativeai as genai
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Vietnamese Law Chatbot", version="1.0.0")

embedder = Embedder(config.EMBEDDING_MODEL, config.USE_GPU, normalize=True)
faiss_index = FaissIndex(config.FAISS_DIR)
try:
    faiss_index.load()
except Exception:
    pass

es = ESClient(config.ES_HOST, config.ES_INDEX, config.ES_USER, config.ES_PASS)
reranker = ReRanker(config.RERANK_MODEL, config.USE_GPU)

if not config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY chưa được thiết lập trong .env!")

genai.configure(api_key=config.GEMINI_API_KEY)


W_SEM = config.HYBRID_W_SEM
W_BM25 = config.HYBRID_W_BM25
@app.get("/")
def root():
    return {"message": "Vietnamese Law Chatbot API is running"}

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
            chunks = hierarchical_chunk(
                it.text,
                parent_id=it.id,
                child_size=config.CHUNK_CHILD_TOKENS if hasattr(config, "CHUNK_CHILD_TOKENS") else 128,
                stride=config.CHUNK_STRIDE if hasattr(config, "CHUNK_STRIDE") else 64
            )
        for ch in chunks:
                cid = ch["child_id"]
                texts.append(ch["text"])
                ids.append(cid)

                meta = {
                    "root_doc": it.id,
                    "parent_id": it.id,
                    "child_id": ch["child_id"],
                    **(it.meta or {}),
                    "text": ch["text"]
                }
                metas.append(meta)
                es_docs.append({
                    "id": cid,
                    "text": ch["text"],
                    "meta": {k: v for k, v in meta.items() if k != "text"}
                })

        embs = embedder.encode_passages(texts)
        faiss_index.add(embs, ids, metas)
        faiss_index.save()
        es.bulk_upsert(es_docs)
        return {"indexed": len(ids)}

@app.post("/search")
def search(req: SearchRequest):
    q = embedder.encode_queries([req.query])
    sem_scores, sem_idxs = faiss_index.index.search(q, max(req.k, 20))
    sem_scores, sem_idxs = sem_scores[0], sem_idxs[0]

    sem_hits = []
    for s, idx in zip(sem_scores, sem_idxs):
        if idx == -1:
            continue
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

    bm_hits = es.search(req.query, k=max(req.k, 20))

    if req.mode == "bm25":
        return {"mode": "bm25", "results": bm_hits}

    bm_scores = [b["score"] for b in bm_hits if "score" in b]
    if bm_scores:
        bm_min, bm_max = min(bm_scores), max(bm_scores)
        for b in bm_hits:
            if "score" in b:
                b["score"] = (b["score"] - bm_min) / (bm_max - bm_min + 1e-8)

    bm_map = {d["doc_id"]: d for d in bm_hits}

    hybrid = []
    seen = set()
    for h in sem_hits:
        bm = bm_map.get(h["doc_id"])
        bm_score = bm["score"] if bm and "score" in bm else 0.0
        final = W_SEM * h["score_semantic"] + W_BM25 * bm_score
        hybrid.append({
            **h,
            "score_bm25": float(bm_score),
            "score_hybrid": float(final)
        })
        seen.add(h["doc_id"])

    for b in bm_hits:
        if b["doc_id"] in seen:
            continue
        final = W_SEM * 0.0 + W_BM25 * b["score"]
        hybrid.append({
            "doc_id": b["doc_id"],
            "text": b["text"],
            "meta": b["meta"],
            "score_semantic": 0.0,
            "score_bm25": float(b["score"]),
            "score_hybrid": float(final)
        })

    hybrid.sort(key=lambda x: x["score_hybrid"], reverse=True)
    hybrid_top = hybrid[:max(req.k, 10)]

    if req.mode == "hybrid":
        return {"mode": "hybrid", "results": hybrid_top[:req.k]}

    reranked = reranker.rerank(req.query, hybrid_top)
    return {"mode": "hybrid_rerank", "results": reranked}

@app.post("/generate")
def generate(payload: dict):
    query = payload.get("query", "")
    k = payload.get("k", 3)

    q_emb = embedder.encode_queries([query])
    scores, idxs = faiss_index.index.search(q_emb, 20)

    sem_hits = []
    for s, i in zip(scores[0], idxs[0]):
        doc_id = faiss_index.id_map[i]
        meta = faiss_index.meta_map[doc_id]
        sem_hits.append({"doc_id": doc_id, "text": meta["text"], "score_semantic": float(s)})

    bm_hits = es.search(query, k=20) if es else []
    bm_map = {d["doc_id"]: d for d in bm_hits}

    # Chuẩn hóa BM25 scores về [0, 1]
    bm_scores = [b["score"] for b in bm_hits if "score" in b]
    if bm_scores:
        bm_min, bm_max = min(bm_scores), max(bm_scores)
        for b in bm_hits:
            if "score" in b:
                b["score_norm"] = (b["score"] - bm_min) / (bm_max - bm_min + 1e-8)
    else:
        bm_min, bm_max = 0, 1

    bm_map = {d["doc_id"]: d for d in bm_hits}
    pool = []
    for h in sem_hits:
        b = bm_map.get(h["doc_id"])
        bm_score = b["score_norm"] if b and "score_norm" in b else 0.0
        hybrid_score = W_SEM * h["score_semantic"] + W_BM25 * bm_score
        pool.append({
            **h,
            "score_bm25": float(bm_score),
            "score_hybrid": float(hybrid_score)
        })

    pool.sort(key=lambda x: x["score_hybrid"], reverse=True)
    top_k = pool[:k]

    context = "\n".join([f"- {c['text']}" for c in top_k])

    prompt = f"""
Bạn là một trợ lý pháp lý thông minh, chuyên trả lời câu hỏi dựa trên các đoạn luật tiếng Việt.
Dưới đây là các đoạn văn bản pháp luật liên quan:

{context}

Câu hỏi: {query}

Hãy trả lời chính xác, ngắn gọn, và dẫn chiếu điều luật nếu có thể.
Nếu không đủ thông tin, hãy nói rõ "Không chắc chắn dựa trên dữ liệu hiện có."
"""

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        answer = response.text.strip() if hasattr(response, "text") else str(response)
    except Exception as e:
        return {"error": f"Lỗi khi gọi Gemini API: {e}"}

    return {
        "query": query,
        "answer": answer,
        "contexts": top_k
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
