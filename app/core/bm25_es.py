from elasticsearch import Elasticsearch, helpers
from typing import Iterable, List, Dict, Any
import ujson as json
class ESClient:
    def __init__(self, host: str, index: str, user: str | None = None, pwd: str | None = None):
        self.es = Elasticsearch(
            hosts=[host],
            basic_auth=(user, pwd) if user and pwd else None
        )
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
                    "text": {"type": "text", "analyzer": "standard"},
                    "meta": {
                        "properties": {
                            "parent_id": {"type": "keyword"},
                            "law_name": {"type": "text"},
                            "article": {"type": "text"},
                            "query_example": {"type": "text"},
                            "chunk_index": {"type": "integer"},
                            "total_chunks": {"type": "integer"}
                        }
                    }
                }
            }
        }
        self.es.indices.create(index=self.index, body=body)
        print(f"Created new index: {self.index}")

    def bulk_upsert(self, docs: List[Dict[str, Any]]):
        """Đưa dữ liệu vào Elasticsearch"""
        actions = [
            {
                "_op_type": "index",
                "_index": self.index,
                "_id": d["id"],
                "_source": {
                    "id": d["id"],  
                    "text": d["text"],
                    "meta": d.get("meta", {})
                },
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

        results = []
        for h in hits:
            src = h["_source"]
            results.append({
                "doc_id": src.get("id", h.get("_id")),  # lấy id an toàn
                "score": float(h["_score"]),
                "text": src.get("text", ""),
                "law_name": src.get("meta", {}).get("law_name"),
                "article": src.get("meta", {}).get("article"),
                "query_example": src.get("meta", {}).get("query_example"),
                "chunk_index": src.get("meta", {}).get("chunk_index"),
                "total_chunks": src.get("meta", {}).get("total_chunks"),
            })
        return results
