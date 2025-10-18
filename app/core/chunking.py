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
  
    toks = text.split()
    out, start, cid = [], 0, 0
    while start < len(toks):
        end = min(start + child_size, len(toks))
        chunk = " ".join(toks[start:end]).strip()
        if chunk:
            out.append({
                "parent_id": parent_id,
                "child_id": f"{parent_id}_c{cid}",
                "text": chunk
            })
        if end == len(toks):
            break
        start = max(0, end - stride)
        cid += 1
    return out
