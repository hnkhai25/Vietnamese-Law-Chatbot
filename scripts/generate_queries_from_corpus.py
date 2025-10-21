import ujson as json

input_path = "data/corpus.jsonl"        
output_path = "data/queries.jsonl"      

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        item = json.loads(line)
        query = item.get("meta", {}).get("query_example", "").strip()
        doc_id = item.get("id")
        if query and doc_id:
            query_item = {"query": query, "answers": [doc_id]}
            fout.write(json.dumps(query_item, ensure_ascii=False) + "\n")

print(f"Generated queries file saved at: {output_path}")
