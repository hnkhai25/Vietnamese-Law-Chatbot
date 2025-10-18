from elasticsearch import Elasticsearch, helpers
from typing import Iterable
import ujson as json
class ESClient:
    def __init__(self, host: str, index: str, user: str | None = None, pwd: str | None = None):
        self.es = Elasticsearch(hosts=[host], basic_auth=(user, pwd) if user and pwd else None)
        self.index = index
        self._ensure_index()

    def _ensure_index(self):
        if self.es.indices.exists(index=self.index):
            return
        body = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "meta": {"type": "object", "enabled": True}
                }
            }
        }
        self.es.indices.create(index=self.index, body=body)

    def bulk_upsert(self, docs):
        actions = [
            {
                "_op_type": "index",
                "_index": self.index,
                "_id": d["id"],
                "_source": {"text": d["text"], "meta": d["meta"]},
            }
            for d in docs
        ]
        try:
            helpers.bulk(self.es, actions)
            print(f"Indexed {len(docs)} docs vào Elasticsearch ({self.index})")
        except Exception as e:
            print("Bulk index failed:")
            if hasattr(e, 'errors'):
                for err in e.errors[:3]:  
                    print(json.dumps(err, indent=2, ensure_ascii=False))
            raise e

    def search(self, query: str, k: int = 10):
        body = {
            "query": {
                "match": {
                    "text": {
                        "query": query,
                        "operator": "and"
                    }
                }
            },
            "size": k
        }
        res = self.es.search(index=self.index, body=body)
        hits = res["hits"]["hits"]
        out = []
        for h in hits:
            src = h["_source"]
            out.append({
                "doc_id": src["id"],
                "score": float(h["_score"]),
                "text": src["text"],
                "meta": src.get("meta", {})
            })
        return out
