import ujson as json, uuid, os
from app.settings import settings
from app.core.embedding import Embedder
from app.core.chunking import simple_chunk
from app.core.index_faiss import FaissIndex
from app.core.bm25_es import ESClient

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    embedder = Embedder(settings.EMBEDDING_MODEL, settings.USE_GPU, True)
    faiss_index = FaissIndex(settings.FAISS_DIR)
    es = ESClient(settings.ES_HOST, settings.ES_INDEX, settings.ES_USER, settings.ES_PASS)

    texts, ids, metas, es_docs = [], [], [], []

    for row in read_jsonl("data/corpus.jsonl"):
        rid = row.get("id") or str(uuid.uuid4())
        text = row["text"]
        meta = row.get("meta", {})

        chunks = simple_chunk(text, settings.CHUNK_MAX_TOKENS, settings.CHUNK_STRIDE)
        for i, ch in enumerate(chunks):
            cid = f"{rid}::chunk:{i}"
            ids.append(cid)
            texts.append(ch)
            m = {"parent_id": rid, "chunk_id": i, **meta, "text": ch}
            metas.append(m)
            es_docs.append({"id": cid, "text": ch, "meta": {k:v for k,v in m.items() if k!="text"}})

    embs = embedder.encode_passages(texts)
    faiss_index.add(embs, ids, metas)
    faiss_index.save()
    es.bulk_upsert(es_docs)
    print(f"Indexed {len(ids)} chunks.")

if __name__ == "__main__":
    main()
