"""
predict.py - Wrapper inference (dùng khi đi thi với BTC)
Nạp Optimized Prompt → LLM sinh python_code → exec() → xuất JSON chuẩn BTC.

Sử dụng:
    uv run python src/predict.py --query "Tính năng lượng tụ C=100uF, U=30V"
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dspy
import config
from solver import PhysicsSolver
from metric import clean_python_code, execute_code
from translator import translate_vi_to_en, preload as preload_translator


def setup_lm():
    import urllib.request
    has_vllm = False
    try:
        req = urllib.request.Request(f"{config.LM_API_BASE}/models", headers={"Authorization": f"Bearer {config.LM_API_KEY}"})
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                has_vllm = True
    except Exception:
        has_vllm = False

    if has_vllm:
        lm = dspy.LM(
            model=f"openai/{config.BASE_MODEL_ID}",
            api_base=config.LM_API_BASE,
            api_key=config.LM_API_KEY,
            max_tokens=1024,
            temperature=0.1,
        )
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        lm = dspy.LM(
            model=f"huggingface/{config.BASE_MODEL_ID}",
            model_type="chat",
            device_map="auto",
            torch_dtype="float16",
            max_tokens=1024,
            temperature=0.1,
        )

    dspy.configure(lm=lm)

    # Preload translator model vào RAM/VRAM ngay khi khởi động
    if config.USE_TRANSLATION:
        preload_translator()


def load_optimized_solver() -> PhysicsSolver:
    """Nạp module đã được tối ưu từ file JSON."""
    if not os.path.exists(config.OPTIMIZED_PROMPT_PATH):
        raise FileNotFoundError(
            f"Chưa tìm thấy Optimized Prompt tại: {config.OPTIMIZED_PROMPT_PATH}\n"
            "Hãy chạy 'uv run python src/optimize.py' trước!"
        )
    solver = PhysicsSolver()
    solver.load(config.OPTIMIZED_PROMPT_PATH)
    print(f"[Predict] Đã nạp Optimized Prompt từ: {config.OPTIMIZED_PROMPT_PATH}")
    return solver


def _format_final_answer(final_answer) -> str:
    """
    Chuẩn hóa final_answer thành chuỗi theo chuẩn BTC.
    - Số đơn: 0.045000001 → "0.045"
    - List   : [0.5, 1.25] → "0.5; 1.25"
    """
    if isinstance(final_answer, (list, tuple)):
        parts = []
        for v in final_answer:
            try:
                parts.append(_format_number(float(v)))
            except (ValueError, TypeError):
                parts.append(str(v))
        return "; ".join(parts)

    try:
        return _format_number(float(final_answer))
    except (ValueError, TypeError):
        return str(final_answer).strip()


def _format_number(val: float) -> str:
    """
    Format số float thành chuỗi gọn:
    - Không trailing zero không cần thiết
    - Dùng ký hiệu khoa học nếu quá nhỏ/quá lớn
    """
    if val == 0:
        return "0"
    abs_val = abs(val)
    if abs_val < 1e-4 or abs_val >= 1e6:
        # Ký hiệu khoa học: 4.5e-5 → "4.5 * 10^-5"
        formatted = f"{val:.4e}"
        # Chuẩn hoá: 4.5000e-05 → 4.5e-5
        mantissa, exp = formatted.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp_int = int(exp)
        return f"{mantissa} * 10^{exp_int}"
    else:
        # Số thường: bỏ trailing zero
        formatted = f"{val:.6f}".rstrip("0").rstrip(".")
        return formatted


def _build_reasoning_steps(extraction: str, python_code: str) -> list[str]:
    """
    Ghép extraction + từng dòng code có nghĩa thành reasoning.steps.
    Đây là điểm ăn P3 (Reasoning Depth) của BTC.
    """
    steps = []

    # Bước 1: Dữ kiện từ extraction
    if extraction and extraction.strip():
        steps.append(f"[Dữ kiện] {extraction.strip()}")

    # Bước 2-3: Từng dòng code có nghĩa (bỏ dòng trống)
    for line in python_code.splitlines():
        line = line.strip()
        if not line:
            continue
        steps.append(line)

    return steps


def solve(solver: PhysicsSolver, query_id: str, query: str) -> dict:
    """
    Wrapper chính:
    1. (Tùy chọn) Dịch câu hỏi Vi→En bằng VinAI translate
    2. Đưa câu hỏi vào PhysicsSolver → nhận extraction + python_code thô
    3. Làm sạch code → exec() → lấy final_answer, unit
    4. Ghép JSON chuẩn BTC
    """
    # --- Bước 1: Dịch Vi → En (nếu USE_TRANSLATION = True) ---
    if config.USE_TRANSLATION:
        question_for_llm = translate_vi_to_en(query)
    else:
        question_for_llm = query

    # --- Bước 2: LLM sinh code (nhận tiếng Anh, sinh python_code) ---
    pred = solver(question=question_for_llm)
    raw_code   = str(pred.python_code)
    extraction = str(pred.extraction).strip()

    # --- Wrapper: làm sạch và chạy code ---
    clean_code = clean_python_code(raw_code)
    final_answer, unit = execute_code(clean_code)

    # --- Xử lý fallback nếu exec() thất bại ---
    if final_answer is None:
        answer_str = "ERROR"
        unit = ""
        explanation = (
            f"[Lỗi] Không thể thực thi mã Python sinh ra. "
            f"Dữ kiện: {extraction}"
        )
        steps = [extraction, "[Lỗi thực thi code]", raw_code]
    else:
        answer_str  = _format_final_answer(final_answer)
        explanation = (
            f"{extraction} "
            f"Tính bằng Python: final_answer = {answer_str} {unit}."
        ).strip()
        steps = _build_reasoning_steps(extraction, clean_code)

    # --- Ghép JSON theo đúng định dạng BTC ---
    result = {
        "query_id":    query_id,
        "answer":      answer_str,
        "unit":        unit if unit else "",
        "explanation": explanation,
        "reasoning": {
            "type":  "cot",
            "steps": steps,
        }
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViPRA 2026 - PAL Solver")
    parser.add_argument("--query", type=str, required=True, help="Nội dung đề bài")
    parser.add_argument("--id",    type=str, default="TEST001", help="Query ID")
    args = parser.parse_args()

    setup_lm()
    solver = load_optimized_solver()

    print(f"\n[Query] {args.query}\n")
    result = solve(solver, query_id=args.id, query=args.query)

    output = json.dumps([result], ensure_ascii=False, indent=2)
    print("[Output JSON]")
    print(output)
