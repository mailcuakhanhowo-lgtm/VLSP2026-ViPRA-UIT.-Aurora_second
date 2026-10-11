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
from data_loader import load_train, load_dev
from solver import PhysicsSolver
from metric import btc_correctness_metric
from translator import translate_batch, preload as preload_translator


def setup_lm():
    """Khởi tạo kết nối tới Language Model (vLLM local)."""
    print(f"[Setup] Kết nối tới model: {config.BASE_MODEL_ID}")
    print(f"[Setup] API Base: {config.LM_API_BASE}")

    lm = dspy.LM(
        model=f"openai/{config.BASE_MODEL_ID}",
        api_base=config.LM_API_BASE,
        api_key=config.LM_API_KEY,
        max_tokens=1024,
        temperature=0.1,
    )
    dspy.configure(lm=lm)
    print("[Setup] Kết nối thành công!\n")
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

        # Tạo lại Examples với câu hỏi đã dịch sang tiếng Anh
        train_data = [
            dspy.Example(
                question=q_en,
                answer=ex.answer,
                unit=ex.unit,
            ).with_inputs("question")
            for ex, q_en in zip(train_data, train_questions_en)
        ]
        dev_data = [
            dspy.Example(
                question=q_en,
                answer=ex.answer,
                unit=ex.unit,
            ).with_inputs("question")
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
    optimizer = MIPROv2(
        metric=btc_correctness_metric,
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
    # trainset: DSPy dùng để bootstrap + sinh instruction candidates
    # valset  : DSPy dùng để đánh giá và chọn Prompt tốt nhất (quan trọng!)
    # num_trials: Số vòng tìm kiếm Bayesian - tăng lên để tốt hơn nhưng chậm hơn
    optimized_module = optimizer.compile(
        student=student_module,
        trainset=train_data,
        valset=dev_data,
        num_trials=config.MIPRO_NUM_TRIALS,
        minibatch=True,               # Dùng minibatch để tối ưu nhanh hơn
        minibatch_size=min(25, len(train_data)),
        minibatch_full_eval_steps=5,  # Cứ 5 trial thì đánh giá full trên dev
        requires_permission_to_run=False,  # Tự chạy không hỏi xác nhận
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
    print(f"\n[Result] Điểm P1 (Correctness) cuối cùng trên Dev: {final_score:.2%}")

    # --- 5. Lưu kết quả ---
    os.makedirs(config.DSPY_OUTPUT_DIR, exist_ok=True)
    optimized_module.save(config.OPTIMIZED_PROMPT_PATH)
    print(f"\n[Save] Đã lưu Optimized Prompt tại: {config.OPTIMIZED_PROMPT_PATH}")
    print("[Done] Hoàn tất! Bây giờ anh có thể dùng file này trong API server.\n")

    return optimized_module, final_score


if __name__ == "__main__":
    # Bước 1: Kết nối LM
    setup_lm()

    # Bước 2: Chạy tối ưu và lưu kết quả
    optimized_module, score = run_optimization()
