import requests
import json

API_URL = "http://localhost:8000/generate"

def test_generate():
    payload = {
        "query": "Công dân có được gia hạn đăng ký tạm trú nhiều lần không?",
        "k": 3
    }

    print(f"🔹 Gửi request đến {API_URL}")
    print("Payload:", json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        response = requests.post(API_URL, json=payload, timeout=300)
    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối: {e}")
        return

    if response.status_code != 200:
        print(f"API trả về mã lỗi {response.status_code}")
        print(response.text)
        return

    data = response.json()
    print("\nKết quả phản hồi:")
    print(f"--- Câu hỏi ---\n{data.get('query')}")
    print(f"\n--- Trả lời ---\n{data.get('answer')}\n")

    if "contexts" in data:
        print("--- Top context ---")
        for i, ctx in enumerate(data["contexts"], 1):
            print(f"{i}. [{ctx.get('doc_id')}] (score={ctx.get('score_hybrid', 0):.4f})")
            print(ctx.get("text", "")[:200] + "...\n")

if __name__ == "__main__":
    test_generate()
