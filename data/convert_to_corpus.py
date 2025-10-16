import json
import uuid

input_file = "data\QA_law_data.json"     
output_file = "data\corpus.jsonl"   
def convert_to_corpus(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  

    with open(output_file, "w", encoding="utf-8") as out:
        for i, item in enumerate(data):
            doc_id = f"doc_{i+1}"
            text = item.get("context", "").strip()
            query_example = item.get("query", "").strip()

            if not text:
                continue  

            corpus_item = {
                "id": doc_id,
                "text": text,
                "meta": {
                    "query_example": query_example
                }
            }

            out.write(json.dumps(corpus_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_to_corpus(input_file, output_file)
