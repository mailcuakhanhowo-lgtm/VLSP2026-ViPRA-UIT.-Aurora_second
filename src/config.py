import os
import glob

# ==========================================
# 1. PATH & DATA CONFIGURATIONS (Auto-detect Kaggle)
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
RAG_DIR = os.path.join(BASE_DIR, "rag_db")

# Tự động nhận diện môi trường Kaggle
KAGGLE_INPUT_DIR = "/kaggle/input"
if os.path.exists(KAGGLE_INPUT_DIR):
    # Dùng recursive=True để lục tìm file ở bất kỳ độ sâu thư mục nào
    train_files = glob.glob(f"{KAGGLE_INPUT_DIR}/**/train_dataset.csv", recursive=True)
    dev_files = glob.glob(f"{KAGGLE_INPUT_DIR}/**/dev_dataset.csv", recursive=True)
    
    RAW_TRAIN_CSV = train_files[0] if train_files else ""
    RAW_DEV_CSV = dev_files[0] if dev_files else ""
    RAW_VAL_CSV = "" # Val nội bộ Kaggle thường gom chung
else:
    # Chạy trên máy tính Local
    RAW_TRAIN_CSV = os.path.join(DATA_DIR, "train_dataset.csv")
    RAW_VAL_CSV = os.path.join(DATA_DIR, "val_dataset.csv")
    RAW_DEV_CSV = os.path.join(DATA_DIR, "dev_dataset.csv")

# --- File Processed ---
PROCESSED_TRAIN_JSONL = os.path.join(DATA_DIR, "train_processed.jsonl")
PROCESSED_HOLDOUT_JSONL = os.path.join(DATA_DIR, "holdout_val.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ==========================================
# 2. BASE MODEL CONFIGURATION (LLM chính)
# ==========================================
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ==========================================
# 3. RAG CONFIGURATION (Hybrid Search)
# ==========================================
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda"
EMBEDDING_DIMENSION = 1024

VECTOR_DB_BACKEND = "faiss"                 
FAISS_INDEX_PATH = os.path.join(RAG_DIR, "faiss_index.bin")
BM25_INDEX_PATH = os.path.join(RAG_DIR, "bm25_index.pkl")
RAG_METADATA_PATH = os.path.join(RAG_DIR, "metadata.pkl")

BM25_TOP_K = 20
DENSE_TOP_K = 20
RRF_K = 60
FINAL_TOP_K = 2
SIMILARITY_THRESHOLD = 0.75

# Bật/Tắt chế độ chạy thử (Dev/Test). Đặt None để chạy toàn bộ dữ liệu.
JUST_TEST_QUERIES = 5

# ==========================================
# 4. PROMPT TEMPLATES (Qwen2.5)
# ==========================================
SYSTEM_PROMPT = """Bạn là một Chuyên gia Vật lý xuất sắc. Nhiệm vụ của bạn là giải các bài toán Vật lý phức tạp một cách chính xác.
Luật cốt lõi (BẮT BUỘC TUÂN THỦ):
1. Trình bày bài giải MẠCH LẠC qua đúng 4 bước: (1) Nhận diện dữ kiện, (2) Công thức, (3) Thay số và tính toán, (4) Kết luận.
2. Bạn KHÔNG BAO GIỜ được chào hỏi, KHÔNG dạ thưa, KHÔNG dùng các cụm từ như "Dưới đây là...", "Vâng...". Hãy đi thẳng vào nội dung giải bài.
3. Sử dụng ĐỘC QUYỀN định dạng mã LaTeX cho mọi công thức toán học và vật lý."""

OUTPUT_FORMAT_REMINDER = """Yêu cầu về format cuối cùng (Bắt buộc):
Sau khi giải xong 4 bước ở trên, hãy tự động chốt đáp án ở 2 dòng cuối cùng với đúng định dạng:
ANSWER: <chỉ ghi 1 con số đáp án>
UNIT: <chỉ ghi đơn vị của đáp án>"""

ASSISTANT_PREFILL = "Sau đây là câu trả lời chi tiết với 4 phần theo đúng yêu cầu:\n\n1. Nhận diện dữ kiện:"
