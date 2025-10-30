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

**Mục tiêu:**
Biến kho dữ liệu pháp luật Việt Nam thành hệ thống tìm kiếm và trả lời tự động đáng tin cậy - hỗ trợ người dân và luật sư tra cứu nhanh chóng, chính xác và minh bạch.


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
| Docker            | ≥ 25.0                |

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

## 4. Chuẩn bị dữ liệu

### 4.1. Chuyển đổi dữ liệu Q&A thành corpus

```bash
python data/convert_to_corpus.py
```

### 4.2. Tạo FAISS index và nạp vào Elasticsearch

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

## 5. Chạy Backend (FastAPI)

### 5.1. Chạy trực tiếp bằng Uvicorn

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5.2. Kiểm tra API

* Health check: [http://localhost:8000/health](http://localhost:8000/health)
* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 6. Chạy Frontend (React)

### 6.1. Cài đặt

```bash
cd frontend
npm install
```

### 6.2. Chạy dev server

```bash
npm start
```

Ứng dụng chạy tại: [http://localhost:3000](http://localhost:3000)

---

## 7. Các API chính

| Method | Endpoint  | Mô tả                                               |
| ------ | --------- | --------------------------------------------------- |
| POST   | /index    | Nạp dữ liệu corpus vào FAISS & Elasticsearch        |
| POST   | /search   | Tìm kiếm (semantic / bm25 / hybrid / hybrid_rerank) |
| POST   | /generate | Sinh câu trả lời bằng Gemini API                    |
| GET    | /health   | Kiểm tra trạng thái server                          |

---
## 8. Demo

Giao diện minh họa hoạt động của hệ thống **Vietnamese Law Chatbot**:

![Demo](assets\image.png)

*Hình minh họa quá trình truy vấn và sinh câu trả lời dựa trên RAG pipeline.*
