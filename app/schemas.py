from pydantic import BaseModel
from typing import Any, Dict, List, Literal

class IndexItem(BaseModel):
    id: str
    text: str
    meta: Dict[str, Any] | None = None

class IndexRequest(BaseModel):
    items: List[IndexItem]
    chunk: bool = True

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    mode: Literal["semantic", "bm25", "hybrid", "hybrid_rerank"]

