"""
optimize_hyde.py - Tối ưu hóa Prompt mở rộng truy vấn (HyDE) độc lập bằng DSPy MIPROv2
Mục tiêu:
    - Nhận đề bài vật lý tiếng Việt
    - Tự động sinh ra các định luật, thuật ngữ và công thức vật lý bằng TIẾNG ANH
    - Giúp kết nối ngữ nghĩa giữa đề thi tiếng Việt và kho bài giảng Feynman tiếng Anh
    - Đánh giá trực tiếp bằng độ tương đồng Cosine Similarity kéo về từ RAG (BGE-M3 + FAISS)
Kết quả lưu tại: output/optimized_hyde.json

Sử dụng:
    uv run python src/optimize_hyde.py
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dspy
from dspy.teleprompt import MIPROv2

import config
from optimize_data_loader import load_train, load_dev


# ============================================================
# BƯỚC 1: SIGNATURE & MODULE DÀNH CHO HYDE
# ============================================================
class GenerateHyDESignature(dspy.Signature):
    """
    Analyze the given physics problem (which may be in Vietnamese).
    Identify the core electromagnetism principles, laws, and algebraic formula names in ENGLISH.
    Output ONLY a concise list of English physics terminology and formula names. Do not solve the problem.
    """
    question: str = dspy.InputField(desc="The physics problem statement (can be in Vietnamese).")
    physics_keywords: str = dspy.OutputField(
        desc="Concise English physics laws, principles, and formulas (e.g., Ohm's Law, Joule heating, P = I^2*R)."
    )


class PhysicsHyDE(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(GenerateHyDESignature)

    def forward(self, question: str):
        return self.generator(question=question)


# ============================================================
# BƯỚC 2: HÀM METRIC ĐÁNH GIÁ CHẤT LƯỢNG TỪ KHÓA HYDE
# Đo trực tiếp độ tương đồng ngữ nghĩa (Cosine Similarity) kéo về từ RAG
# ============================================================
_rag_evaluator = None

def get_rag_evaluator():
    """Khởi tạo RAG Pipeline dạng Singleton để chấm điểm nhanh chóng."""
    global _rag_evaluator
    if _rag_evaluator is None:
        from rag_module import RAGPipeline
        rag = RAGPipeline()
        rag.load_indices()
        _rag_evaluator = rag
    return _rag_evaluator


def hyde_retrieval_gain_metric(example: object, pred: object, trace=None) -> float:
    """
    Metric khách quan cho HyDE:
    1. Kiểm tra định dạng: Ngắn gọn, không giải bài, không sinh CoT.
    2. Chạy Semantic Search với truy vấn mở rộng f"{question} {keywords}".
    3. Điểm số tỷ lệ thuận với Cosine Similarity đạt được với bài giảng Feynman liên quan.
    """
    keywords = str(getattr(pred, "physics_keywords", "")).strip()

    # 1. Ràng buộc cơ bản (Hard Constraints)
    if not keywords or len(keywords) < 8:
        return 0.0

    # Phạt nặng nếu mô hình sa đà vào giải bài hoặc sinh code Python
    forbidden_markers = ["def ", "final_answer", "import ", "1. nhận diện", "bước 1", "giải:"]
    if any(m in keywords.lower() for m in forbidden_markers):
        return 0.1

    # Phạt nếu quá dài (> 50 từ) - HyDE chỉ cần từ khóa và công thức súc tích
    words = keywords.split()
    length_penalty = 1.0 if len(words) <= 50 else 0.5

    # 2. Thưởng điểm nếu có công thức đại số hoặc ký hiệu vật lý chuẩn
    has_formula = bool(re.search(r"[A-Za-z0-9_]+\s*=\s*[A-Za-z0-9_+\-*/^]+", keywords))
    formula_bonus = 0.15 if has_formula else 0.0

    # 3. Đo lường hiệu quả truy xuất thực tế với BGE-M3 + FAISS
    try:
        rag = get_rag_evaluator()
        # Mở rộng câu hỏi với từ khóa HyDE
        expanded_query = f"{example.question} {keywords}"
        dense_results = rag.semantic_search(expanded_query, top_k=2)

        if dense_results:
            top_sim = float(dense_results[0]["score"])
            # Chuẩn hóa điểm Cosine Similarity về thang [0.0, 0.85]
            # Điểm Cosine >= 0.70 được xem là xuất sắc
            retrieval_score = min(0.85, max(0.0, (top_sim - 0.3) / 0.45))
        else:
            retrieval_score = 0.3
    except Exception:
        retrieval_score = 0.4

    total_score = (retrieval_score + formula_bonus) * length_penalty
    return round(min(1.0, max(0.0, total_score)), 3)


# ============================================================
# BƯỚC 3: SETUP LM & CHẠY OPTIMIZATION
# ============================================================
def setup_hyde_lm():
    """Khởi tạo LM kết nối tới vLLM (DeepSeek-Math-7B)."""
    import urllib.request
    has_vllm = False
    try:
        req = urllib.request.Request(f"{config.LM_API_BASE}/models", headers={"Authorization": f"Bearer {config.LM_API_KEY}"})
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                has_vllm = True
    except Exception:
        has_vllm = False

    vllm_model_id = config.BASE_MODEL_ID
    if has_vllm:
        import json
        try:
            req = urllib.request.Request(f"{config.LM_API_BASE}/models", headers={"Authorization": f"Bearer {config.LM_API_KEY}"})
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    if data.get("data") and len(data["data"]) > 0:
                        vllm_model_id = data["data"][0]["id"]
        except Exception:
            pass

        print(f"[Setup HyDE] Kết nối vLLM server: {config.LM_API_BASE} (Model: {vllm_model_id})")
        lm = dspy.LM(
            model=f"openai/{vllm_model_id}",
            api_base=config.LM_API_BASE,
            api_key=config.LM_API_KEY,
            max_tokens=128,
            temperature=0.1,
        )
    else:
        print(f"[Setup HyDE] Nạp model {config.BASE_MODEL_ID} qua Transformers cục bộ...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        hf_model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        hf_tok = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
        pipe = pipeline("text-generation", model=hf_model, tokenizer=hf_tok, max_new_tokens=128, do_sample=False)

        class LocalLM(dspy.LM):
            def __init__(self, p, name):
                super().__init__(model=name)
                self.pipe = p
            def basic_request(self, prompt, **kwargs): pass
            def __call__(self, prompt=None, messages=None, **kwargs):
                text = prompt if prompt else messages[-1]["content"]
                out = self.pipe(text, max_new_tokens=128, do_sample=False)
                return [out[0]["generated_text"][len(text):].strip()]

        lm = LocalLM(pipe, config.BASE_MODEL_ID)

    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    print("[Setup HyDE] Mô hình đã sẵn sàng!\n")
    return lm


def run_hyde_optimization():
    """
    Quy trình tối ưu Prompt cho HyDE:
    1. Load tập Train & Dev
    2. MIPROv2 tìm kiếm Instruction tiếng Anh tối ưu và Bootstrap Few-shot demos
    3. Lưu ra output/optimized_hyde.json
    """
    print("=" * 60)
    print("  KHỞI ĐỘNG OPTIMIZER CHO HYDE (DEEPSEEK-MATH-7B)")
    print("  Mục tiêu: Tối ưu hóa trích xuất từ khóa tiếng Anh cho RAG Feynman")
    print("=" * 60)

    train_data = load_train()
    dev_data = load_dev()

    hyde_trainset = [
        dspy.Example(question=ex.question).with_inputs("question")
        for ex in train_data[:min(50, len(train_data))]
    ]
    hyde_devset = [
        dspy.Example(question=ex.question).with_inputs("question")
        for ex in dev_data[:min(15, len(dev_data))]
    ]

    print(f"[HyDE Optimizer] Train size: {len(hyde_trainset)} | Dev size: {len(hyde_devset)}\n")

    student_module = PhysicsHyDE()

    optimizer = MIPROv2(
        metric=hyde_retrieval_gain_metric,
        auto=None,
        num_candidates=5,
        max_bootstrapped_demos=3,
        max_labeled_demos=0,
        verbose=True,
    )

    print("[HyDE Optimizer] Bắt đầu tối ưu hóa Prompt...\n")
    optimized_hyde = optimizer.compile(
        student=student_module,
        trainset=hyde_trainset,
        valset=hyde_devset,
        num_trials=15,
        minibatch=True,
        minibatch_size=min(15, len(hyde_trainset)),
    )

    # Đánh giá lần cuối trên Dev set
    print("\n[Evaluate HyDE] Đánh giá kết quả cuối cùng trên Dev set...")
    evaluator = dspy.Evaluate(
        devset=hyde_devset,
        metric=hyde_retrieval_gain_metric,
        num_threads=1,
        display_progress=True,
        display_table=3,
    )
    final_score = evaluator(optimized_hyde)
    try:
        score_val = float(final_score) if not hasattr(final_score, 'score') else float(final_score.score)
        print(f"\n[Result] Điểm HyDE Retrieval cuối cùng trên Dev: {score_val:.2%}")
    except Exception:
        print(f"\n[Result] Đánh giá hoàn tất: {final_score}")

    # Lưu kết quả
    os.makedirs(config.DSPY_OUTPUT_DIR, exist_ok=True)
    optimized_hyde.save(config.OPTIMIZED_HYDE_PATH)
    print(f"\n[Save] Đã lưu Optimized HyDE tại: {config.OPTIMIZED_HYDE_PATH}")
    return optimized_hyde


if __name__ == "__main__":
    setup_hyde_lm()
    run_hyde_optimization()
