Rõ rồi ✅
Dưới đây là **phiên bản `README.md` hoàn chỉnh không có emoji hoặc icon**, chuẩn Markdown để bạn **copy thẳng vào file `README.md`** của dự án:

---

# Vietnamese Law Chatbot (RAG System)

**Vietnamese-Law-Chatbot** là hệ thống Retrieval-Augmented Generation (RAG) giúp trả lời các câu hỏi pháp luật tiếng Việt.
Hệ thống kết hợp:

* FAISS (Semantic Search)
* Elasticsearch (BM25)
* Gemini API (Google Generative AI)
  để sinh ra câu trả lời chính xác, có trích dẫn điều luật.

---

## Mục tiêu dự án

* Tự động trả lời câu hỏi pháp lý tiếng Việt từ nguồn văn bản luật chính thống.
* Cho phép tìm kiếm luật thông minh (hybrid semantic + lexical).
* Sinh câu trả lời tự nhiên và có trích dẫn bằng Gemini LLM.
* Cung cấp API backend (FastAPI) và giao diện web frontend (React).

---

## 1. Cấu trúc thư mục

```bash
VietNamese-Law-Chatbot/
├── app/
│   ├── main.py
│   ├── core/
│   ├── schemas.py
│   ├── settings.py
├── data/
├── indices/
├── frontend/
├── scripts/
├── test/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 2. Yêu cầu hệ thống

| Thành phần        | Phiên bản khuyến nghị |
| ----------------- | --------------------- |
| Python            | ≥ 3.10                |
| Node.js           | ≥ 18.0                |
| Elasticsearch     | 8.x                   |
| FAISS             | CPU hoặc GPU          |
| pip               | ≥ 24.0                |
| Docker (tùy chọn) | ≥ 25.0                |

---

## 3. Cài đặt môi trường thủ công

### 3.1. Clone project

```bash
git clone https://github.com/yourusername/Vietnamese-Law-Chatbot.git
cd Vietnamese-Law-Chatbot
```

### 3.2. Tạo virtual environment

```bash
python -m venv venv
venv\Scripts\activate     # Windows
# hoặc
source venv/bin/activate  # Linux/macOS
```

### 3.3. Cài dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Cấu hình môi trường (.env)

Tạo file `.env` trong thư mục gốc:

```env
# ===== EMBEDDING / SEARCH =====
EMBEDDING_MODEL=BAAI/bge-m3
USE_GPU=False

# ===== FAISS =====
FAISS_DIR=indices/faiss

# ===== Elasticsearch =====
ES_HOST=http://localhost:9200
ES_INDEX=law_corpus
ES_USER=elastic
ES_PASS=changeme

# ===== RERANK =====
RERANK_MODEL=vinai/phobert-base
HYBRID_W_SEM=0.7
HYBRID_W_BM25=0.3

# ===== Gemini API =====
GEMINI_API_KEY=your_google_gemini_api_key_here
```

Lấy key Gemini tại: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## 5. Chuẩn bị dữ liệu

### 5.1. Chuyển đổi dữ liệu Q&A thành corpus

```bash
python data/convert_to_corpus.py
```

### 5.2. Tạo FAISS index và nạp vào Elasticsearch

```bash
python scripts/index_corpus.py
```

Sau khi chạy xong, thư mục `indices/faiss/` sẽ chứa:

```
faiss.index
embeddings.npy
id_map.json
meta_map.json
```

---

## 6. Chạy Backend (FastAPI)

### 6.1. Chạy trực tiếp bằng Uvicorn

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6.2. Kiểm tra API

* Health check: [http://localhost:8000/health](http://localhost:8000/health)
* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 7. Chạy Frontend (React)

### 7.1. Cài đặt

```bash
cd frontend
npm install
```

### 7.2. Chạy dev server

```bash
npm start
```

Ứng dụng chạy tại: [http://localhost:3000](http://localhost:3000)

---

## 8. Deploy bằng Docker (Production)

### 8.1. Cấu trúc Dockerfile

```Dockerfile
FROM python:3.10-slim AS backend
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app ./app
COPY ./data ./data
COPY ./indices ./indices
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2. docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./indices:/app/indices
      - ./data:/app/data
    depends_on:
      - elasticsearch

  frontend:
    build:
      context: ./frontend
    ports:
      - "3000:3000"
    command: ["npm", "start"]
    depends_on:
      - backend

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.9.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data

volumes:
  es_data:
```

---

## 9. Đánh giá hiệu năng (Evaluate)

```bash
python scripts/evaluate_vlc.py --queries data/queries.jsonl
```

Ví dụ kết quả:

```
SEMANTIC
Recall@1: 0.84
Recall@3: 0.91
MRR: 0.88

HYBRID
Recall@1: 0.89
Recall@3: 0.94
MRR: 0.90
```

---

## 10. Các API chính

| Method | Endpoint  | Mô tả                                               |
| ------ | --------- | --------------------------------------------------- |
| POST   | /index    | Nạp dữ liệu corpus vào FAISS & Elasticsearch        |
| POST   | /search   | Tìm kiếm (semantic / bm25 / hybrid / hybrid_rerank) |
| POST   | /generate | Sinh câu trả lời bằng Gemini API                    |
| GET    | /health   | Kiểm tra trạng thái server                          |

---

## 11. Giao diện Frontend

Frontend React gồm hai chế độ:

* Search Mode: hiển thị top-k văn bản luật liên quan
* Chat Mode: người dùng hỏi và nhận câu trả lời từ Gemini

Các API được gọi đến:

* [http://localhost:8000/search](http://localhost:8000/search)
* [http://localhost:8000/generate](http://localhost:8000/generate)

---

## 12. Troubleshooting

| Lỗi                       | Nguyên nhân              | Cách khắc phục                          |
| ------------------------- | ------------------------ | --------------------------------------- |
| KeyError: 'id'            | Corpus thiếu `id`        | Kiểm tra lại `convert_to_corpus.py`     |
| OSError: 1455             | Thiếu RAM khi load model | Dùng model nhỏ hơn                      |
| Gemini API not found      | Sai model name           | Dùng `models/gemini-2.5-flash`          |
| npx not recognized        | Node.js chưa cài         | Cài Node.js và thêm vào PATH            |
| Cannot find file: App.css | Sai tên file             | Đổi `app.css → App.css` hoặc sửa import |

---

## 14. Liên hệ

Nếu bạn muốn đóng góp hoặc báo lỗi, vui lòng mở issue hoặc pull request trên GitHub.
Email: [hoangngockhai000@gmail.com](mailto:hoangngockhai000@gmail.com)

---

**Mục tiêu:**
Biến kho dữ liệu pháp luật Việt Nam thành hệ thống tìm kiếm và trả lời tự động đáng tin cậy — hỗ trợ người dân và luật sư tra cứu nhanh chóng, chính xác và minh bạch.

