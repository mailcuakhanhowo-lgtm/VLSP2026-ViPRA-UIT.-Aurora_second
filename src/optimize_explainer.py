"""
optimize_explainer.py - Tối ưu hóa Prompt sinh Lời giải thích (Explanation) cho Qwen2.5-7B
Sử dụng phương pháp:
    - Kết hợp G-Eval Rubric (Tiêu chuẩn đánh giá không cần nhãn ground-truth)
    - DSPy MIPROv2 với LLM-as-a-Judge đánh giá tính sư phạm và công thức LaTeX
Kết quả lưu tại: output/optimized_explainer.json

Sử dụng độc lập:
    python src/optimize_explainer.py
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShotWithRandomSearch

import config
from optimize_data_loader import load_train, load_dev
from result_explainer import PhysicsExplainer, PhysicsExplainerSignature


# ============================================================
# BƯỚC 1: ĐỊNH NGHĨA GIÁM KHẢO G-EVAL (LLM-as-a-Judge)
# ============================================================
class GEvalExplanationJudge(dspy.Signature):
    """
    Bạn là Giám khảo chấm thi Vật lý chuyên nghiệp theo tiêu chuẩn G-Eval (Evaluation Rubric).
    Nhiệm vụ: Đánh giá chất lượng của 'explanation' (lời giải thích vật lý bằng tiếng Việt) dựa trên 3 tiêu chí cốt lõi:
    
    1. Súc tích & Tự nhiên (Conciseness): Lời giải thích ngắn gọn, tự nhiên trong 1 đến 3 câu.
       TUYỆT ĐỐI KHÔNG chia 4 bước dài dòng kiểu CoT (1. Nhận diện, 2. Công thức, 3. Thay số, 4. Kết luận).
       KHÔNG có câu chào hỏi thừa như "Dưới đây là...".
       
    2. Chuẩn xác Vật lý & Cơ sở lý thuyết (Faithfulness & Physics Foundation):
       Trình bày rõ bản chất vật lý, cơ sở lý thuyết và tên định luật / nguyên lý áp dụng (tham khảo RAG context).
       Có công thức ĐẠI SỐ TỔNG QUÁT bằng mã LaTeX ($...$).
       TUYỆT ĐỐI KHÔNG sa đà vào tự thế số trung gian hay tính toán số học (để tránh mâu thuẫn số liệu với đáp án đã cho).
       
    3. Nhất quán với Đáp số (Consistency):
       Lời giải thích logic, kết luận phù hợp với đáp án mục tiêu {target_answer} {target_unit}.
    """
    question: str = dspy.InputField(desc="Đề bài câu hỏi vật lý.")
    context: str = dspy.InputField(desc="Tài liệu lý thuyết RAG liên quan.")
    target_answer: str = dspy.InputField(desc="Đáp án cuối cùng chuẩn của bài toán.")
    target_unit: str = dspy.InputField(desc="Đơn vị vật lý chuẩn.")
    explanation: str = dspy.InputField(desc="Lời giải thích do mô hình Explainer sinh ra cần chấm điểm.")

    reasoning: str = dspy.OutputField(desc="Nhận xét ngắn gọn 1 câu theo tiêu chí G-Eval.")
    score: float = dspy.OutputField(desc="Điểm số float từ 0.0 đến 1.0 (1.0 là xuất sắc, 0.5 là trung bình, 0.0 là sai/vô nghĩa).")


# ============================================================
# BƯỚC 2: HÀM METRIC TỔNG HỢP: G-EVAL RUBRIC + RULE-BASED FILTER
# ============================================================
def geval_explanation_metric(example: object, pred: object, trace=None) -> float:
    """
    Hàm chấm điểm kết hợp Phương pháp 1 (DSPy LLM-as-a-Judge) và Phương pháp 3 (G-Eval Rubric):
    - Đánh giá không cần nhãn ground-truth text.
    - Phạt nặng nếu sinh CoT 4 bước dài dòng hoặc rỗng.
    - Thưởng điểm cho công thức LaTeX và điểm đánh giá của Giám khảo LLM.
    """
    explanation = str(getattr(pred, "explanation", "")).strip()

    # 1. Bộ lọc nhanh cơ bản (Hard Constraints)
    if not explanation or len(explanation) < 20:
        return 0.0

    # Phạt nặng nếu mô hình sinh lặp lại CoT 4 bước dài dòng thay vì 1 đoạn văn giải thích súc tích
    cot_markers = ["1. nhận diện dữ kiện", "2. công thức", "3. thay số và tính toán", "4. kết luận"]
    has_cot_structure = sum(1 for m in cot_markers if m in explanation.lower()) >= 2
    if has_cot_structure:
        return 0.1  # Vi phạm yêu cầu ngắn gọn của explanation

    # Phạt nếu quá dài (> 120 từ)
    word_count = len(explanation.split())
    if word_count > 120:
        length_penalty = 0.5
    else:
        length_penalty = 1.0

    # 2. Kiểm tra định dạng LaTeX (Rule-based: 20% điểm)
    has_latex = bool(re.search(r"\$.*?\$|\\(?:frac|cdot|times|sqrt|omega|mu|pi)", explanation))
    format_score = 1.0 if has_latex else 0.2

    # 3. G-Eval LLM-as-a-Judge chấm nội dung (80% điểm)
    target_ans = str(getattr(example, "answer", "")).strip()
    target_u = str(getattr(example, "unit", "")).strip()
    ctx = str(getattr(example, "context", config.RAG_FALLBACK_TEXT))

    try:
        judge = dspy.Predict(GEvalExplanationJudge)
        eval_result = judge(
            question=str(example.question),
            context=ctx,
            target_answer=target_ans,
            target_unit=target_u,
            explanation=explanation,
        )
        score_raw = str(getattr(eval_result, "score", "0.5"))
        match = re.search(r"(\d+(?:\.\d+)?)", score_raw)
        if match:
            s_val = float(match.group(1))
            if s_val > 1.0 and s_val <= 10.0:
                s_val /= 10.0
            elif s_val > 10.0:
                s_val /= 100.0
            judge_score = max(0.0, min(1.0, s_val))
        else:
            judge_score = 0.5
    except Exception:
        # Fallback an toàn nếu LLM-judge gặp sự cố
        judge_score = 0.5

    # Điểm tổng hợp cuối cùng
    total_score = (0.2 * format_score + 0.8 * judge_score) * length_penalty
    return round(min(1.0, max(0.0, total_score)), 3)


# ============================================================
# BƯỚC 3: SETUP LM & CHẠY OPTIMIZATION
# ============================================================
def setup_explainer_lm():
    """Khởi tạo LM cho Qwen2.5-7B (Explainer)."""
    import urllib.request
    has_vllm = False
    try:
        req = urllib.request.Request(
            f"{config.EXPLAINER_API_BASE}/models",
            headers={"Authorization": f"Bearer {config.EXPLAINER_API_KEY}"}
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                has_vllm = True
    except Exception:
        has_vllm = False

    explainer_model_id = config.EXPLAINER_MODEL_ID
    if has_vllm:
        import json
        try:
            req = urllib.request.Request(
                f"{config.EXPLAINER_API_BASE}/models",
                headers={"Authorization": f"Bearer {config.EXPLAINER_API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    if data.get("data") and len(data["data"]) > 0:
                        detected_model = data["data"][0]["id"]
                        if detected_model != config.EXPLAINER_MODEL_ID:
                            print(f"[Cảnh báo Explainer] vLLM đang chạy model: {detected_model}")
                            print(f"[Cảnh báo Explainer] Nhưng config ghi: {config.EXPLAINER_MODEL_ID}")
                            print(f"[Setup Explainer] Tự động đồng bộ dùng model đang chạy: {detected_model}")
                            explainer_model_id = detected_model
        except Exception:
            pass

        print(f"[Setup Explainer] Kết nối vLLM server: {config.EXPLAINER_API_BASE}")
        print(f"[Setup Explainer] Kết nối tới model: {explainer_model_id}")
        lm = dspy.LM(
            model=f"openai/{explainer_model_id}",
            api_base=config.EXPLAINER_API_BASE,
            api_key=config.EXPLAINER_API_KEY,
            max_tokens=512,
            temperature=0.2,
        )
    else:
        print(f"[Setup Explainer] Tải model {config.EXPLAINER_MODEL_ID} qua Transformers cục bộ...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        hf_model = AutoModelForCausalLM.from_pretrained(
            config.EXPLAINER_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(config.EXPLAINER_MODEL_ID, trust_remote_code=True)
        pipe = pipeline(
            "text-generation",
            model=hf_model,
            tokenizer=hf_tokenizer,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.2,
        )

        class LocalTransformersLM(dspy.LM):
            def __init__(self, pipe, model_name):
                super().__init__(model=model_name)
                self.pipe = pipe

            def basic_request(self, prompt, **kwargs):
                pass

            def __call__(self, prompt=None, messages=None, **kwargs):
                if messages is not None:
                    prompt_text = self.pipe.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_text = prompt
                out = self.pipe(prompt_text, max_new_tokens=kwargs.get("max_tokens", 512), do_sample=False)
                gen = out[0]["generated_text"]
                return [gen[len(prompt_text):].strip() if gen.startswith(prompt_text) else gen.strip()]

        lm = LocalTransformersLM(pipe, config.EXPLAINER_MODEL_ID)

    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    print("[Setup Explainer] Mô hình đã sẵn sàng!\n")
    return lm


def run_explainer_optimization():
    """
    Quy trình tối ưu hóa Prompt cho Explainer:
    1. Load train/dev dataset (lấy question, context, answer, unit làm đầu vào)
    2. Khởi tạo PhysicsExplainer module
    3. Dùng MIPROv2 tối ưu Instruction bằng hàm geval_explanation_metric
    4. Lưu kết quả ra output/optimized_explainer.json
    """
    print("=" * 60)
    print("  KHỞI ĐỘNG OPTIMIZER CHO EXPLAINER (QWEN2.5-7B)")
    print("  Phương pháp: DSPy MIPROv2 + G-Eval Rubric Metric")
    print("=" * 60)

    train_data = load_train()
    dev_data = load_dev()

    # Trích xuất lý thuyết RAG gắn vào context (Dữ liệu cố định)
    try:
        from rag_module import RAGPipeline
        rag = RAGPipeline()
        rag.load_indices()
        for ex in train_data:
            ex.context = rag.get_context(ex.question)
        for ex in dev_data:
            ex.context = rag.get_context(ex.question)
        del rag
    except Exception:
        pass

    # Chuẩn bị Example với đầu vào đầy đủ: question, context, answer, unit
    explainer_trainset = [
        dspy.Example(
            question=ex.question,
            context=getattr(ex, "context", config.RAG_FALLBACK_TEXT),
            answer=str(ex.answer),
            unit=str(getattr(ex, "unit", "")),
        ).with_inputs("question", "context", "answer", "unit")
        for ex in train_data[:min(60, len(train_data))]  # Tập mẫu tối ưu vừa phải để tiết kiệm thời gian
    ]

    explainer_devset = [
        dspy.Example(
            question=ex.question,
            context=getattr(ex, "context", config.RAG_FALLBACK_TEXT),
            answer=str(ex.answer),
            unit=str(getattr(ex, "unit", "")),
        ).with_inputs("question", "context", "answer", "unit")
        for ex in dev_data[:min(20, len(dev_data))]
    ]

    print(f"[Explainer Optimizer] Train size: {len(explainer_trainset)} | Dev size: {len(explainer_devset)}\n")

    student_module = PhysicsExplainer()

    optimizer = MIPROv2(
        metric=geval_explanation_metric,
        auto=None,
        num_candidates=5,
        max_bootstrapped_demos=2,
        max_labeled_demos=1,
        verbose=True,
    )

    print("[Explainer Optimizer] Bắt đầu tối ưu hóa Prompt...\n")
    optimized_explainer = optimizer.compile(
        student=student_module,
        trainset=explainer_trainset,
        valset=explainer_devset,
        num_trials=15,
        minibatch=True,
        minibatch_size=min(15, len(explainer_trainset)),
    )

    # Lưu kết quả
    os.makedirs(config.DSPY_OUTPUT_DIR, exist_ok=True)
    optimized_explainer.save(config.OPTIMIZED_EXPLAINER_PATH)
    print(f"\n[Save] Đã lưu Optimized Explainer tại: {config.OPTIMIZED_EXPLAINER_PATH}")
    return optimized_explainer


if __name__ == "__main__":
    setup_explainer_lm()
    run_explainer_optimization()
