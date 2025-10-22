def simple_chunk(text: str, max_tokens: int = 256, stride: int = 32) -> list[str]:
    toks = text.split()
    out, start = [], 0
    while start < len(toks):
        end = min(start + max_tokens, len(toks))
        chunk = " ".join(toks[start:end]).strip()
        if chunk:
            out.append(chunk)
        if end == len(toks):
            break
        start = max(0, end - stride)
    return out
def hierarchical_chunk(text: str, parent_id: str, child_size: int = 128, stride: int = 64):
    """Cắt văn bản thành nhiều đoạn có chồng lấn để giữ ngữ cảnh."""
    toks = text.split()
    out, start, cid = [], 0, 0
    while start < len(toks):
        end = min(start + child_size, len(toks))
        chunk = " ".join(toks[start:end]).strip()
        if chunk:
            out.append({
                "parent_id": parent_id,
                "child_id": f"{parent_id}_c{cid}",
                "chunk_index": cid,  
                "text": chunk
            })
        if end == len(toks):
            break
        start = max(0, end - stride)
        cid += 1

    total = len(out)
    for item in out:
        item["total_chunks"] = total
    return out
