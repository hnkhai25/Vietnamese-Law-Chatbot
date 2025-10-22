from app.core.chunking import hierarchical_chunk
import json
import re
import os
from app.settings import config



def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_law_name(text: str) -> str:
    pattern = r"(Luật|Nghị định|Thông tư)\s+[A-ZÂĂÊÔƠƯa-zàáạảãầấậẩẫăằắặẳẵèéẹẻẽêềếệểễòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ\-–/,.\s\d]+"
    match = re.search(pattern, text)
    return match.group(0).strip() if match else ""


def extract_article_info(text: str) -> str:
    pattern = r"(Điều\s*\d+(?:\s*,\s*Khoản\s*\d+)?)"
    match = re.search(pattern, text)
    return match.group(0).strip() if match else ""


def convert_to_corpus(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Không tìm thấy file: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", encoding="utf-8") as out:
        total_docs, total_chunks = 0, 0

        for i, item in enumerate(data):
            parent_id = f"doc_{i+1}"
            query_example = item.get("query", "").strip()
            context = clean_text(item.get("context", ""))

            if not context:
                continue

            law_name = extract_law_name(context)
            article = extract_article_info(context)

            prefix_parts = []
            if law_name:
                prefix_parts.append(law_name)
            if article:
                prefix_parts.append(article)
            prefix = " - ".join(prefix_parts) + ": " if prefix_parts else ""

            chunks = hierarchical_chunk(context, parent_id, child_size=config.CHUNK_CHILD_TOKENS, stride=config.CHUNK_STRIDE)

            for chunk in chunks:
                enriched_text = prefix + chunk["text"]

                corpus_item = {
                    "id": chunk["child_id"],
                    "text": enriched_text,
                    "meta": {
                        "parent_id": parent_id,
                        "law_name": law_name,
                        "article": article,
                        "query_example": query_example,
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": chunk["total_chunks"],
                    }
                }

                out.write(json.dumps(corpus_item, ensure_ascii=False) + "\n")
                total_chunks += 1

            total_docs += 1

    print(f"Đã tạo corpus: {output_file}")
    print(f"   ├── Tổng văn bản gốc: {total_docs}")
    print(f"   └── Tổng số chunk sinh ra: {total_chunks}")


if __name__ == "__main__":
    input_file = "data/QA_law_data.json"
    output_file = "data/corpus.jsonl"
    convert_to_corpus(input_file, output_file)
