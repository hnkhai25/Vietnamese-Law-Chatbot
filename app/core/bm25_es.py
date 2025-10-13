from elasticsearch import Elasticsearch, helpers
from typing import Iterable

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

    def bulk_upsert(self, docs: Iterable[dict]):
        actions = []
        for d in docs:
            actions.append({
                "_op_type": "index",
                "_index": self.index,
                "_id": d["id"],
                "_source": d
            })
        helpers.bulk(self.es, actions)

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
