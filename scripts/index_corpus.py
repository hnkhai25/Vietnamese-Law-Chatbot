import ujson as json, uuid, os
from app.settings import config
from app.core.embedding import Embedder
from app.core.chunking import simple_chunk
from app.core.index_faiss import FaissIndex
from app.core.bm25_es import ESClient
from app.core.chunking import hierarchical_chunk

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    embedder = Embedder(config.EMBEDDING_MODEL, config.USE_GPU, True)
    faiss_index = FaissIndex(config.FAISS_DIR)
    es = ESClient(config.ES_HOST, config.ES_INDEX, config.ES_USER, config.ES_PASS)

    texts, ids, metas, es_docs = [], [], [], []
    total_parents, total_children = 0, 0

    for row in read_jsonl("data/corpus.jsonl"):
        rid = row.get("id") or str(uuid.uuid4())   # ID document (parent)
        text = row["text"]
        meta = row.get("meta", {})

        chunks = hierarchical_chunk(
            text,
            parent_id=rid,
            child_size=config.CHUNK_CHILD_TOKENS if hasattr(config, "CHUNK_CHILD_TOKENS") else 128,
            stride=config.CHUNK_STRIDE if hasattr(config, "CHUNK_STRIDE") else 64
        )

        total_parents += 1
        total_children += len(chunks)

        for ch in chunks:
            cid = ch["child_id"]    
            ids.append(cid)
            texts.append(ch["text"])

            m = {
                "root_doc": rid,       # parent document ID
                "parent_id": rid,
                "child_id": ch["child_id"],
                **meta,
                "text": ch["text"]
            }
            metas.append(m)

            es_docs.append({
                "id": cid,
                "text": ch["text"],
                "meta": {k: v for k, v in m.items() if k != "text"}
            })

    print(f"Encoding {len(texts)} child chunks ...")
    embs = embedder.encode_passages(texts)

    faiss_index.add(embs, ids, metas)
    faiss_index.save()

    print("Indexing into Elasticsearch ...")
    es.bulk_upsert(es_docs)

    print(f"Indexed {total_children} child chunks thuộc {total_parents} parent documents.")


if __name__ == "__main__":
    main()
