import os

# ==========================================
# 1. PATH & DATA CONFIGURATIONS (Đường dẫn & Dữ liệu)
# ==========================================
# File này nằm trong thư mục src, nên BASE_DIR sẽ lùi lại 1 cấp ra thư mục VLSP2026
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset") # Dùng thư mục dataset như đã setup ở .gitignore

# --- File Raw (Do BTC cung cấp) ---
RAW_TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
RAW_VAL_CSV = os.path.join(DATA_DIR, "val.csv") # Tập Hold-out (Thi thử thực chiến)

# --- Thông số Split (Dành cho file train.csv) ---
TRAIN_SPLIT_RATIO = 0.9  # 90% dữ liệu gốc để Train
VAL_SPLIT_RATIO = 0.1    # 10% dữ liệu gốc làm Val nội bộ (Để Kaggle vẽ biểu đồ Loss)
RANDOM_SEED = 42         # Đảm bảo chia data ngẫu nhiên nhưng cố định mỗi lần chạy

# --- File Processed (Sau khi chạy script băm nhỏ thành JSONL) ---
PROCESSED_TRAIN_JSONL = os.path.join(DATA_DIR, "train_90.jsonl")
PROCESSED_VAL_INTERNAL_JSONL = os.path.join(DATA_DIR, "val_10.jsonl")
PROCESSED_HOLDOUT_JSONL = os.path.join(DATA_DIR, "holdout_val.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ==========================================
# 2. BASE MODEL CONFIGURATION (LLM chính)
# ==========================================
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Backup: BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# ==========================================
# 3. RAG CONFIGURATION (Hybrid Search)
# ==========================================
# --- Embedding Model (Dense Retrieval) ---
EMBEDDING_MODEL_ID = "BAAI/bge-m3"          # 569M params, ~2.27GB VRAM, hỗ trợ tiếng Việt
EMBEDDING_DEVICE = "cuda"                   # Để trên VRAM cho tốc độ ~0.05s/query
EMBEDDING_DIMENSION = 1024                  # Chiều vector của BGE-M3

# --- Vector DB ---
VECTOR_DB_BACKEND = "faiss"                 # Hoặc "chromadb"
FAISS_INDEX_PATH = "rag/faiss_index"        # Đường dẫn lưu FAISS index
BM25_INDEX_PATH = "rag/bm25_index.pkl"      # Đường dẫn lưu BM25 index

# --- Retrieval Settings ---
BM25_TOP_K = 20                             # Số kết quả BM25 lấy trước khi RRF
DENSE_TOP_K = 20                            # Số kết quả Semantic lấy trước khi RRF
RRF_K = 60                                  # Hệ số RRF (chuẩn = 60)
FINAL_TOP_K = 2                             # Số bài mẫu cuối cùng nhét vào Prompt

# ==========================================
# 3. LORA / QLORA CONFIGURATION (Tham số Fine-tune)
# ==========================================
USE_4BIT_QUANTIZATION = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ==========================================
# 4. TRAINING HYPERPARAMETERS (Siêu tham số)
# ==========================================
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 1024
OPTIMIZER = "paged_adamw_8bit"
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 50

# ==========================================
# 5. PROMPT TEMPLATE CONFIGURATION (Tối ưu v2)
# ==========================================

# --- System Prompt (Vai trò & 3 Luật cốt lõi) ---
SYSTEM_PROMPT = """\
Bạn là Chuyên gia Vật lý Điện-Từ. Hãy giải bài toán dựa trên phương pháp của các ví dụ mẫu.

3 LUẬT CỨNG BẮT BUỘC:
1. Trình bày đúng 4 bước giải chi tiết (1. Nhận diện, 2. Công thức, 3. Thay số, 4. Kết luận).
2. Công thức toán học sử dụng chuẩn mã LaTeX.
3. KHÔNG ĐƯỢC bịa số liệu, KHÔNG chép lại số liệu từ ví dụ mẫu. Đọc kỹ số liệu đề bài chính.
(Tuyệt đối KHÔNG chào hỏi, dạ thưa hay giải thích dài dòng ở đầu và cuối.)\
"""

# --- Few-Shot Wrapper (Dùng trong RAG Inference) ---
FEW_SHOT_EXAMPLE_TEMPLATE = """\
╔══════════════════════════════════════════════════════╗
║             VÍ DỤ THAM KHẢO SỐ {i}                  ║
╚══════════════════════════════════════════════════════╝
[ĐỀ BÀI] {question}
[LỜI GIẢI]
{cot}
[ĐÁP ÁN] {answer}
[ĐƠN VỊ] {unit}\
"""

# --- Cảnh báo chống học vẹt ---
ANTI_MEMORIZATION_WARNING = """\
⚠️ LƯU Ý: Ví dụ trên chỉ để học CÁCH SUY LUẬN. Đề bài dưới đây có số liệu HOÀN TOÀN KHÁC.\
"""

# --- User Prompt (Đề bài chính) ---
USER_PROMPT_TEMPLATE = """\
{few_shot_examples}

{anti_memorization_warning}

[ĐỀ BÀI CẦN GIẢI]
{question}\
"""

# --- Output Format (Định dạng kết quả mong muốn) ---
# Dùng văn bản thuần, Regex Python sẽ tự động tách và bắt lỗi
OUTPUT_FORMAT_REMINDER = """\
Sau khi giải xong 4 bước, hãy chốt lại đáp án ở 2 dòng cuối cùng như sau:
ANSWER: <con số, nếu nhiều đáp án cách nhau bằng ;>
UNIT: <đơn vị>\
"""

# --- Kỹ thuật Mớm lời (Prefill Prompting) ---
# Sẽ được nối thẳng vào Role 'Assistant' lúc nạp vào vLLM để chặn AI chào hỏi
ASSISTANT_PREFILL = "Sau đây là câu trả lời với 4 phần chi tiết theo đúng yêu cầu:\n\n1. Nhận diện dữ kiện:"
