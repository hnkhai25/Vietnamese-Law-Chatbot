import os, ssl, warnings, requests
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# ⚙️ Tắt SSL verification toàn cục
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
warnings.filterwarnings("ignore", category=UserWarning)
ssl._create_default_https_context = ssl._create_unverified_context

def test_embedding():
    print("🚀 Đang load model all-MiniLM-L6-v2 ...")

    try:
        # 🧠 Tải model mà KHÔNG xác minh SSL
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)
        print("✅ Model tải thành công!")
    except Exception as e:
        print("❌ Hugging Face HTTPS bị lỗi, chuyển sang tải thủ công...")
        try:
            model_path = hf_hub_download(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                filename="config.json",
                local_dir="models/all-MiniLM-L6-v2",
                local_dir_use_symlinks=False,
                force_download=False,
                resume_download=True
            )
            print("✅ Model được tải về:", model_path)
        except Exception as e2:
            print("❌ Vẫn lỗi:", e2)
            return

    sentences = ["Luật Dân sự Việt Nam quy định về quyền và nghĩa vụ của công dân."]
    embeddings = model.encode(sentences)
    print("✅ Sinh vector thành công:", embeddings.shape)

if __name__ == "__main__":
    test_embedding()
