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

# Giới hạn số câu đưa vào trình tối ưu (giảm nếu GPU yếu / test nhanh)
MAX_TRAIN_EXAMPLES = 200
MAX_DEV_EXAMPLES   = 100

# Tham số MIPROv2 - tăng NUM_TRIALS để tối ưu kỹ hơn (nhưng chậm hơn)
MIPRO_NUM_CANDIDATES         = 10  # Số biến thể Instruction được sinh ra
MIPRO_NUM_TRIALS             = 20  # Số vòng tìm kiếm Bayesian
MIPRO_MAX_BOOTSTRAPPED_DEMOS = 3   # Số ví dụ tự bootstrap từ tập train
MIPRO_MAX_LABELED_DEMOS      = 3   # Số ví dụ có nhãn cố định

# ==========================================
# 6. TRANSLATION CONFIGURATION (Vi → En)
# Dùng VinAI translate-vi2en-v2 (~400MB, seq2seq - KHÔNG tính vào giới hạn 8B của BTC)
# ==========================================

# Bật/tắt bước dịch thuật trước khi đưa vào LLM
# True  → Dịch Vi→En (nên dùng khi main model thiên về English)
# False → Giữ nguyên tiếng Việt (nên dùng khi main model hỗ trợ Việt tốt)
USE_TRANSLATION = True

# Model dịch thuật (HuggingFace model ID)
TRANSLATION_MODEL_ID = "vinai/vinai-translate-vi2en-v2"

# Thư mục cache lưu model đã tải về (tránh download lại mỗi lần chạy)
TRANSLATION_CACHE_DIR = os.path.join(BASE_DIR, "models", "vinai-translate")

# Thiết bị chạy translator
# "cuda" → chạy trên GPU (nhanh, ~0.05s/câu, tốn ~400MB VRAM)
# "cpu"  → chạy trên CPU (chậm hơn ~0.3s/câu, không tốn VRAM của main LLM)
TRANSLATION_DEVICE = "cpu"

# Số token tối đa đầu ra của bản dịch (đề vật lý thường < 150 token)
TRANSLATION_MAX_LENGTH = 256

# Batch size khi dịch nhiều câu cùng lúc (dùng trong optimize.py để pre-translate)
TRANSLATION_BATCH_SIZE = 32
