import requests, time

API = "http://localhost:8000"

def test_health():
    r = requests.get(f"{API}/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
if __name__ == "__main__":
    test_health()
    print("All tests passed.")
