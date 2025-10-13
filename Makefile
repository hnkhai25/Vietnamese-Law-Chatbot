.PHONY: up down build index seed test

build:
\tdocker compose build

up:
\tdocker compose up -d

down:
\tdocker compose down -v

index:
\tpython scripts/index_corpus.py

seed:
\tpython - <<'PY'\nimport json, os\nos.makedirs('data', exist_ok=True)\nwith open('data/corpus.jsonl','w',encoding='utf-8') as f:\n f.write(json.dumps({'id':'doc1','text':'BERT là mô hình Transformer cho NLP','meta':{'source':'wiki'}})+'\\n')\n f.write(json.dumps({'id':'doc2','text':'RAG kết hợp truy hồi tài liệu và sinh văn bản','meta':{'source':'blog'}})+'\\n')\n f.write(json.dumps({'id':'doc3','text':'ElasticSearch dùng BM25 để xếp hạng','meta':{'source':'notes'}})+'\\n')\nprint('Seeded data/corpus.jsonl')\nPY

test:
\tpytest -q
