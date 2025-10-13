import os, json
import numpy as np
import faiss

class FaissIndex:
    def __init__(self, dir_path: str):
        self.dir = dir_path
        self.emb = None
        self.id_map: list[str] = []
        self.meta_map: dict[str, dict] = {}
        self.index = None
        self.dim = None

    def add(self, embeddings: np.ndarray, ids: list[str], metas: list[dict]):
        assert len(embeddings) == len(ids) == len(metas)
        if self.emb is None:
            self.emb = embeddings
            self.id_map = ids
        else:
            self.emb = np.vstack([self.emb, embeddings])
            self.id_map.extend(ids)
        for _id, m in zip(ids, metas):
            self.meta_map[_id] = m
        self._build()

    def _build(self):
        if self.emb is None or len(self.emb) == 0:
            return
        self.dim = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.emb)

    def search(self, q: np.ndarray, k: int = 5):
        scores, idxs = self.index.search(q, k)
        return scores[0], idxs[0]

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        np.save(os.path.join(self.dir, "embeddings.npy"), self.emb)
        faiss.write_index(self.index, os.path.join(self.dir, "faiss.index"))
        with open(os.path.join(self.dir, "id_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False)
        with open(os.path.join(self.dir, "meta_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta_map, f, ensure_ascii=False)

    def load(self):
        self.emb = np.load(os.path.join(self.dir, "embeddings.npy"))
        self.index = faiss.read_index(os.path.join(self.dir, "faiss.index"))
        with open(os.path.join(self.dir, "id_map.json"), "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        with open(os.path.join(self.dir, "meta_map.json"), "r", encoding="utf-8") as f:
            self.meta_map = json.load(f)
        self.dim = self.emb.shape[1]
        return self
