"""
optimize.py - Script tối ưu Prompt tự động bằng DSPy MIPROv2
Sử dụng:
    uv run python src/optimize.py
Kết quả được lưu tại: output/optimized_prompt.json
"""
import os
import sys

# Thêm thư mục src vào sys.path để import các module nội bộ
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dspy
from dspy.teleprompt import MIPROv2

import config
from optimize_data_loader import load_train, load_dev
from result_optimize_solver import PhysicsSolver
from result_optimize_metric import btc_correctness_metric
from result_optimize_translator import translate_batch, preload as preload_translator


def setup_lm():
    """Khởi tạo kết nối tới Language Model: tự động chọn vLLM nếu có, hoặc tải thẳng qua Transformers 2x T4."""
    import os
    import urllib.request
    
    # Kiểm tra xem có vLLM server đang chạy ở LM_API_BASE không
    has_vllm = False
    try:
        req = urllib.request.Request(f"{config.LM_API_BASE}/models", headers={"Authorization": f"Bearer {config.LM_API_KEY}"})
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                has_vllm = True
    except Exception:
        has_vllm = False

    if has_vllm:
        print(f"[Setup] Phát hiện vLLM server đang chạy tại: {config.LM_API_BASE}")
        print(f"[Setup] Kết nối tới model: {config.BASE_MODEL_ID}")
        lm = dspy.LM(
            model=f"openai/{config.BASE_MODEL_ID}",
            api_base=config.LM_API_BASE,
            api_key=config.LM_API_KEY,
            max_tokens=512,
            temperature=0.1,
        )
    else:
        print(f"[Setup] Không thấy vLLM server. Chuyển sang nạp trực tiếp qua Transformers...")
        # Chạy Local GPU hoàn toàn (không gọi HuggingFace Inference API đám mây):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        print(f"[Setup] Đang nạp model cục bộ {config.BASE_MODEL_ID} lên 2x T4 GPU (device_map='auto', float16)...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, trust_remote_code=True)
        
        pipe = pipeline(
            "text-generation",
            model=hf_model,
            tokenizer=hf_tokenizer,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.1,
        )

        class LocalTransformersLM(dspy.LM):
            def __init__(self, pipe, model_name):
                super().__init__(model=model_name)
                self.pipe = pipe
                self.model = model_name

            def basic_request(self, prompt, **kwargs):
                pass

            def __call__(self, prompt=None, messages=None, **kwargs):
                # Chuẩn hóa input từ DSPy thành text
                if messages is not None:
                    # Nếu truyền messages dạng chat
                    prompt_text = self.pipe.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_text = prompt

                out = self.pipe(prompt_text, max_new_tokens=kwargs.get("max_tokens", 1024), do_sample=False)
                generated = out[0]["generated_text"]
                # Cắt bỏ phần prompt ban đầu, chỉ lấy phần model sinh ra
                if generated.startswith(prompt_text):
                    response_text = generated[len(prompt_text):].strip()
                else:
                    response_text = generated.strip()
                return [response_text]

        lm = LocalTransformersLM(pipe, config.BASE_MODEL_ID)

    dspy.configure(lm=lm)
    print("[Setup] Mô hình đã sẵn sàng!\n")
    return lm


def run_optimization():
    """
    Luồng chính (PAL):
    1. Load train + dev
    2. DSPy dùng train để bootstrap: LLM sinh python_code → exec() → kiểm tra
       Những câu code chạy đúng đáp án được lưu làm few-shot examples
    3. MIPROv2 tối ưu Instruction + few-shot, đánh giá bằng dev (cô lập)
    4. Lưu optimized_prompt.json
    """
    # --- 1. Load dữ liệu ---
    train_data = load_train()
    dev_data   = load_dev()
    print(f"[Optimizer] Train: {len(train_data)} | Dev: {len(dev_data)}\n")

    # --- 2. Pre-translate Vi → En (1 lần duy nhất, batch cho nhanh) ---
    if config.USE_TRANSLATION:
        preload_translator()
        print("[Optimizer] Pre-translating train set...")
        train_questions_en = translate_batch([ex.question for ex in train_data])
        print("[Optimizer] Pre-translating dev set...")
        dev_questions_en   = translate_batch([ex.question for ex in dev_data])

        # Tạo lại Examples với câu hỏi đã dịch sang tiếng Anh (giữ nguyên context)
        train_data = [
            dspy.Example(
                context=getattr(ex, "context", config.RAG_FALLBACK_TEXT),
                question=q_en,
                answer=ex.answer,
                unit=ex.unit,
            ).with_inputs("context", "question")
            for ex, q_en in zip(train_data, train_questions_en)
        ]
        dev_data = [
            dspy.Example(
                context=getattr(ex, "context", config.RAG_FALLBACK_TEXT),
                question=q_en,
                answer=ex.answer,
                unit=ex.unit,
            ).with_inputs("context", "question")
            for ex, q_en in zip(dev_data, dev_questions_en)
        ]
        print(f"[Optimizer] Translation done. Sample: {train_data[0].question[:80]}...\n")
    else:
        print("[Optimizer] Translation disabled (USE_TRANSLATION=False), using Vietnamese.\n")

    # --- 3. Khởi tạo module ban đầu (chưa tối ưu) ---
    student_module = PhysicsSolver()
    print("[Optimizer] Khởi tạo MIPROv2...")

    # MIPROv2 là bộ tối ưu mạnh nhất của DSPy:
    #   - Tự động sinh biến thể câu lệnh (Instruction) từ tập Train
    #   - Tự động bootstrap Few-shot examples từ tập Train
    #   - Đánh giá và chọn lọc bằng tập Dev (valset) - KHÔNG dùng Dev để sinh ví dụ
    # Trong DSPy 3.x: cần đặt auto=None nếu tự định nghĩa num_candidates & num_trials
    optimizer = MIPROv2(
        metric=btc_correctness_metric,
        auto=None,                                   # Tắt chế độ auto để dùng custom candidates/trials
        num_candidates=config.MIPRO_NUM_CANDIDATES,  # Số biến thể Instruction được sinh ra
        max_bootstrapped_demos=config.MIPRO_MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=config.MIPRO_MAX_LABELED_DEMOS,
        verbose=True,  # Hiển thị tiến trình chi tiết
    )

    print("[Optimizer] Bắt đầu tối ưu Prompt...\n")
    print("=" * 60)
    print("  Train set -> Nguyên liệu sinh Prompt & Few-shot examples")
    print("  Dev set   -> Thước đo đánh giá chốt hạ (KHÔNG bị rò rỉ)")
    print("=" * 60)
    print()

    # --- 3. Chạy tối ưu ---
    optimized_module = optimizer.compile(
        student=student_module,
        trainset=train_data,
        valset=dev_data,
        num_trials=config.MIPRO_NUM_TRIALS,
        minibatch=True,               # Dùng minibatch để tối ưu nhanh hơn
        minibatch_size=min(30, len(train_data)),
        minibatch_full_eval_steps=5,  # Cứ 5 trial thì đánh giá full trên dev
    )

    # --- 4. Đánh giá lần cuối trên Dev set ---
    print("\n[Evaluate] Đánh giá kết quả cuối cùng trên Dev set...")
    evaluator = dspy.Evaluate(
        devset=dev_data,
        metric=btc_correctness_metric,
        num_threads=1,   # vLLM local thường dùng 1 thread
        display_progress=True,
        display_table=5, # Hiển thị 5 ví dụ mẫu đầu tiên
    )
    final_score = evaluator(optimized_module)
    
    # --- 5. Lưu kết quả ---
    os.makedirs(config.DSPY_OUTPUT_DIR, exist_ok=True)
    optimized_module.save(config.OPTIMIZED_PROMPT_PATH)
    print(f"\n[Save] Đã lưu Optimized Prompt tại: {config.OPTIMIZED_PROMPT_PATH}")

    try:
        score_val = float(final_score) if not hasattr(final_score, 'score') else float(final_score.score)
        print(f"\n[Result] Điểm P1 (Correctness) cuối cùng trên Dev: {score_val:.2%}")
    except Exception:
        print(f"\n[Result] Đánh giá hoàn tất: {final_score}")

    print("[Done] Hoàn tất! Bây giờ anh có thể dùng file này trong API server.\n")
    return optimized_module, final_score


if __name__ == "__main__":
    # Bước 1: Kết nối LM
    setup_lm()

    # Bước 2: Chạy tối ưu và lưu kết quả
    optimized_module, score = run_optimization()
