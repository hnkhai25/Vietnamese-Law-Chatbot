from pathlib import Path
from pydantic_settings import BaseSettings

class ApplicationConfig(BaseSettings):
    EMBEDDING_MODEL: str
    RERANK_MODEL: str

    HYBRID_W_SEM: float
    HYBRID_W_BM25: float
    
    CHUNK_CHILD_TOKENS : int
    CHUNK_MAX_TOKENS: int
    CHUNK_STRIDE : int

    FAISS_DIR: str

    ES_HOST: str
    ES_INDEX: str
    ES_USER: str | None = None
    ES_PASS: str | None = None

    API_HOST: str
    API_PORT: int


    USE_GPU: bool

    class Config:
        env_file = "env.example"
        env_file_encoding = "utf-8"


config = ApplicationConfig()
Path(config.FAISS_DIR).mkdir(parents=True, exist_ok=True)
