import os

# ==========================================
# 1. PATH CONFIGURATIONS (Đường dẫn)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_formatted.jsonl")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val_formatted.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ==========================================
# 2. BASE MODEL CONFIGURATION
# ==========================================
# Chọn 1 trong 2 model tối ưu nhất dưới 8B hiện nay
# BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ==========================================
# 3. LORA / QLORA CONFIGURATION (Tham số Fine-tune)
# ==========================================
USE_4BIT_QUANTIZATION = True  # Nén 4-bit để train trên card VRAM thấp (12GB-24GB)
LORA_R = 16                   # Rank: Độ lớn của ma trận adapter (Thường dùng 8 hoặc 16)
LORA_ALPHA = 32               # Alpha = 2 * R là công thức phổ biến nhất
LORA_DROPOUT = 0.05
# Target modules cho họ Llama/Qwen để tune toàn bộ cơ bắp của model
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ==========================================
# 4. TRAINING HYPERPARAMETERS (Siêu tham số)
# ==========================================
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3          # 3 epochs là chuẩn cho Instruction Tuning
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 1024         # Giới hạn chiều dài ngữ cảnh (Bài tập vật lý thường ngắn)
OPTIMIZER = "paged_adamw_8bit" # Tối ưu hóa bộ nhớ
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 50

# ==========================================
# 5. PROMPT TEMPLATE CONFIGURATION
# ==========================================
SYSTEM_PROMPT = """Bạn là một trợ lý AI chuyên giải bài tập Vật lý (chuyên đề Điện - Từ học).
Nhiệm vụ của bạn là đọc đề bài và đưa ra lời giải bắt buộc tuân theo định dạng 4 bước:
1. Nhận diện dữ kiện
2. Công thức
3. Thay số và tính toán
4. Kết luận
Tất cả biểu thức toán học phải được định dạng sạch sẽ, chuẩn xác."""
