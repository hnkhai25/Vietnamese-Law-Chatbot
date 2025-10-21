import requests
import argparse
import json


API_URL = "http://127.0.0.1:5000/search"

def send_search_request(query: str, top_k: int = 5, mode: str = "hybrid"):
    
    payload = {
        "query": query,
        "top_k": top_k,
        "mode": mode
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()  # báo lỗi nếu status != 200
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return

    try:
        result = response.json()
    except json.JSONDecodeError:
        print("Lỗi parse JSON từ response.")
        print(response.text)
        return

    print("Search result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send search query to API.")
    parser.add_argument("--query", type=str, required=True, help="Câu hỏi cần tìm kiếm.")
    parser.add_argument("--top_k", type=int, default=5, help="Số kết quả muốn lấy.")
    parser.add_argument("--mode", type=str, default="hybrid", help="Chế độ tìm kiếm: semantic | bm25 | hybrid | hybrid_rerank")
    parser.add_argument("--api_url", type=str, default=API_URL, help="Địa chỉ API backend.")
    args = parser.parse_args()

    API_URL = args.api_url
    send_search_request(args.query, args.top_k, args.mode)
