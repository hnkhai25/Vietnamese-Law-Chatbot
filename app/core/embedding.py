from sentence_transformers import SentenceTransformer
import numpy as np

def get_instruction_prefixes(model_name: str):
    name = model_name.lower()
    prefixes = {"query": "", "passage": ""}
    if "e5" in name:
        prefixes = {"query": "query: ", "passage": "passage: "}
    elif "bge" in name:
        prefixes = {
            "query": "Represent this sentence for searching relevant passages: ",
            "passage": "Represent this sentence for retrieving relevant passages: "
        }
    return prefixes

class Embedder:
    def __init__(self, model_name: str, use_gpu: bool = False, normalize: bool = True):
        device = "cuda" if use_gpu else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.prefix = get_instruction_prefixes(model_name)

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        texts = [self.prefix["passage"] + t for t in texts]
        return self.model.encode(
            texts, convert_to_numpy=True,
            normalize_embeddings=self.normalize, batch_size=batch_size
        )

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        texts = [self.prefix["query"] + t for t in texts]
        return self.model.encode(
            texts, convert_to_numpy=True,
            normalize_embeddings=self.normalize, batch_size=batch_size
        )
