from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    HYBRID_W_SEM: float = 0.7
    HYBRID_W_BM25: float = 0.3

    FAISS_DIR: str = "indices/faiss"

    ES_HOST: str = "http://localhost:9200"
    ES_INDEX: str = "corpus"
    ES_USER: str | None = None
    ES_PASS: str | None = None

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    CHUNK_MAX_TOKENS: int = 256
    CHUNK_STRIDE: int = 32

    USE_GPU: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
Path(settings.FAISS_DIR).mkdir(parents=True, exist_ok=True)
