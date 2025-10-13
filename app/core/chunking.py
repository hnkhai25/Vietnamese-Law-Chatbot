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
