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
BASE_MODEL_ID = "deepseek-ai/deepseek-math-7b-instruct"

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

# --- Feynman Theory Data & Chunking Configuration ---
FEYNMAN_RAW_DIR = os.path.join(BASE_DIR, "data", "processed", "feynman_vol_II")
PROCESSED_THEORY_JSONL = os.path.join(DATA_DIR, "feynman_theory_chunks.jsonl")

# Thông số Chunking chuyên biệt cho lý thuyết vật lý
CHUNK_TARGET_WORDS = 350       # Số từ mục tiêu cho mỗi chunk (~450-550 tokens)
CHUNK_MIN_WORDS = 100          # Số từ tối thiểu để tạo 1 chunk độc lập
CHUNK_MAX_WORDS = 500          # Ngưỡng tối đa, vượt quá sẽ ngắt đoạn an toàn
CHUNK_OVERLAP_WORDS = 60       # Độ gối đầu giữa các chunk để bảo toàn ngữ cảnh
RAG_FALLBACK_TEXT = "Không có tài liệu tham khảo bổ sung. Hãy sử dụng các định luật vật lý tiêu chuẩn."

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

ASSISTANT_PREFILL = "Sau đây là câu trả lời chi tiết với 4 phần theo đúng yêu cầu:\n"

# ==========================================
# 5. DSPy PROMPT OPTIMIZER CONFIGURATION
# ==========================================
# Đường dẫn lưu Prompt đã tối ưu (output của optimize.py)
DSPY_OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OPTIMIZED_PROMPT_PATH = os.path.join(DSPY_OUTPUT_DIR, "optimized_prompt.json")

# Cấu hình Language Model cho DSPy (trỏ tới vLLM local)
LM_API_BASE = "http://localhost:8000/v1"  # Endpoint vLLM local
LM_API_KEY  = "none"                      # vLLM không cần key thật
# LM_MODEL_ID dùng lại BASE_MODEL_ID đã khai báo ở trên

# Sai số số học cho phép theo luật BTC
NUMERICAL_TOLERANCE = 0.05

# Cấu hình RAG Fallback & Ngưỡng tin cậy
RAG_SIMILARITY_THRESHOLD = 0.75
RAG_FALLBACK_TEXT = "Không có tài liệu tham khảo bổ sung. Hãy sử dụng các định luật vật lý tiêu chuẩn."

# Giới hạn thời gian Sandbox an toàn (tránh treo vô tận)
SANDBOX_TIMEOUT_SECONDS = 15.0

# Giới hạn số câu đưa vào trình tối ưu (giảm nếu GPU yếu / test nhanh)
MAX_TRAIN_EXAMPLES = 200
MAX_DEV_EXAMPLES   = 100

# Tham số MIPROv2 - Cân bằng tối ưu: 10 candidates, 40 trials
MIPRO_NUM_CANDIDATES         = 10  # 10 biến thể Instruction
MIPRO_NUM_TRIALS             = 40  # 40 vòng Bayesian Search
MIPRO_MAX_BOOTSTRAPPED_DEMOS = 3   # 3 ví dụ PAL giải đúng tự động
MIPRO_MAX_LABELED_DEMOS      = 2   # 2 ví dụ có nhãn tham chiếu

# ==========================================
# 6. TRANSLATION CONFIGURATION (Vi → En)
# Dùng VinAI translate-vi2en-v2 (~400MB, seq2seq - KHÔNG tính vào giới hạn 8B của BTC)
# ==========================================

USE_TRANSLATION = True
TRANSLATION_MODEL_ID = "vinai/vinai-translate-vi2en-v2"

# Thư mục cache lưu model đã tải về
if os.path.exists(KAGGLE_INPUT_DIR):
    TRANSLATION_CACHE_DIR = "/kaggle/working/models/vinai-translate"
    # Đảm bảo đường dẫn output nằm trong /kaggle/working/ (Writable)
    DSPY_OUTPUT_DIR = "/kaggle/working/output"
    OPTIMIZED_PROMPT_PATH = os.path.join(DSPY_OUTPUT_DIR, "optimized_prompt.json")
else:
    TRANSLATION_CACHE_DIR = os.path.join(BASE_DIR, "models", "vinai-translate")

# Tự động chia tải GPU (Tối ưu 2x T4 Kaggle):
# Nếu có 2 GPU (Kaggle T4 x2): Đẩy translator sang GPU 1 (cuda:1) để GPU 0 rảnh tải cho LLM
import torch
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        TRANSLATION_DEVICE = "cuda:1"
    else:
        TRANSLATION_DEVICE = "cuda:0"
else:
    TRANSLATION_DEVICE = "cpu"

TRANSLATION_MAX_LENGTH = 256
TRANSLATION_BATCH_SIZE = 32


# ==========================================
# 7. SCRAPER CONFIGURATION (Feynman Lectures & Raw Data)
# ==========================================
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Danh sách URL cần cào dữ liệu (hỗ trợ nhiều URL, mỗi URL tạo 1 thư mục riêng)
SCRAPER_TARGET_URLS = [
    "https://www.feynmanlectures.caltech.edu/II_toc.html",
]

# Headers giả lập trình duyệt để tránh bị máy chủ Caltech chặn 403 Forbidden
SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.feynmanlectures.caltech.edu/",
}

SCRAPER_DELAY = 1.0       # Thời gian nghỉ (giây) giữa các request
SCRAPER_TIMEOUT = 30      # Timeout mỗi request (giây)

# Cấu hình chuẩn hóa ký hiệu toán học (Notation Normalization)
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
NORMALIZER_INPUT_DIR = os.path.join(DATA_RAW_DIR, "feynman_vol_II")
NORMALIZER_OUTPUT_DIR = os.path.join(DATA_PROCESSED_DIR, "feynman_vol_II")

