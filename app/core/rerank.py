from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self, model_name: str, use_gpu: bool = False):
        device = "cuda" if use_gpu else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: list[dict], top_k: int = 5):
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs).tolist()
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return docs[:top_k]
