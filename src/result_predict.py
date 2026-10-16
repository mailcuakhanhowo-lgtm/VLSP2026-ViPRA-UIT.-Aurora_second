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
from result_optimize_solver import PhysicsSolver
from result_optimize_metric import clean_python_code, execute_code
from result_optimize_translator import translate_vi_to_en, preload as preload_translator


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
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        print(f"[Predict Setup] Nạp {config.BASE_MODEL_ID} cục bộ lên GPU (device_map='auto', float16)...")
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
                if messages is not None:
                    prompt_text = self.pipe.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_text = prompt

                out = self.pipe(prompt_text, max_new_tokens=kwargs.get("max_tokens", 1024), do_sample=False)
                generated = out[0]["generated_text"]
                if generated.startswith(prompt_text):
                    response_text = generated[len(prompt_text):].strip()
                else:
                    response_text = generated.strip()
                return [response_text]

        lm = LocalTransformersLM(pipe, config.BASE_MODEL_ID)

    try:
        adapter = dspy.ChatAdapter()
        dspy.configure(lm=lm, adapter=adapter)
        print("[Predict Setup] Đã kích hoạt ChatAdapter để ép sinh text/code chuẩn.")
    except Exception:
        dspy.configure(lm=lm)

    # Preload translator model vào RAM/VRAM ngay khi khởi động (cả 2 chiều vi2en và en2vi để suy luận mượt mà)
    if config.USE_TRANSLATION:
        preload_translator(direction="both")


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


def _build_reasoning_steps(python_code: str) -> list[str]:
    """
    Ghép từng dòng code có nghĩa thành reasoning.steps.
    Thể hiện thuần túy các bước khai báo biến, đổi đơn vị SI và công thức đại số (P3).
    """
    steps = []
    for line in python_code.splitlines():
        line = line.strip()
        if not line:
            continue
        steps.append(line)
    return steps


def solve(solver: PhysicsSolver, query_id: str, query: str, context: str = None) -> dict:
    """
    Wrapper chính:
    1. (Tùy chọn) Dịch câu hỏi Vi→En bằng VinAI translate
    2. Đưa context và câu hỏi vào PhysicsSolver → nhận extraction, explanation_vi (lý thuyết thuần), python_code
    3. Làm sạch code → exec() trong Sandbox (timeout 15s) → lấy final_answer, unit
    4. Ghép JSON chuẩn BTC (P1 số chuẩn, P2 lý thuyết không số, P3 reasoning steps)
    """
    # --- Bước 1: Dịch Vi → En (nếu USE_TRANSLATION = True) ---
    if config.USE_TRANSLATION:
        question_for_llm = translate_vi_to_en(query)
    else:
        question_for_llm = query

    pred = solver(question=question_for_llm, context=context)
    raw_code = str(getattr(pred, "python_code", ""))

    # --- Bước 3: Wrapper: làm sạch và chạy code trong Sandbox có timeout 15s ---
    clean_code = clean_python_code(raw_code)
    final_answer, unit = execute_code(clean_code)

    # --- Bước 4: Xử lý kết quả & Explanation (DeepSeek-Math-7B) ---
    if final_answer is None:
        answer_str = "ERROR"
        unit = ""
        try:
            from result_explainer import generate_explanation
            explanation = generate_explanation(
                question=query,
                context=context,
                answer="chưa xác định",
                unit="",
                question_en=question_for_llm if config.USE_TRANSLATION else None,
            )
        except Exception:
            explanation = "Áp dụng các định luật vật lý tiêu chuẩn trong Điện và Từ học để suy luận và giải bài toán."
        steps = ["[Lỗi thực thi mã Python giải toán]", raw_code] if raw_code else ["[Không sinh được mã thực thi]"]
    else:
        answer_str = _format_final_answer(final_answer)
        try:
            from result_explainer import generate_explanation
            explanation = generate_explanation(
                question=query,
                context=context,
                answer=answer_str,
                unit=unit if unit else "",
                question_en=question_for_llm if config.USE_TRANSLATION else None,
            )
        except Exception:
            explanation = ""

        if not explanation:
            explanation = f"Áp dụng định luật vật lý tính toán: final_answer = {answer_str} {unit}.".strip()

        # reasoning.steps: thể hiện rõ ràng các bước thực thi từ code Python
        steps = _build_reasoning_steps(clean_code)

    # --- Ghép JSON theo đúng định dạng BTC 5 trường ---
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
    parser.add_argument("--query",   type=str, required=True, help="Nội dung đề bài")
    parser.add_argument("--id",      type=str, default="TEST001", help="Query ID")
    parser.add_argument("--use_rag", action="store_true", default=True, help="Bật tra cứu RAG context tự động")
    args = parser.parse_args()

    setup_lm()
    solver = load_optimized_solver()

    context = None
    if args.use_rag:
        try:
            from rag_module import RAGPipeline
            rag = RAGPipeline()
            rag.load_indices()
            context = rag.get_context(args.query)
            print(f"[Predict RAG] Đã nạp context lý thuyết ({len(context)} ký tự).")
        except Exception as e:
            print(f"[Predict RAG] Không nạp được RAG DB ({e}). Tự động dùng fallback context.")
            context = None

    print(f"\n[Query] {args.query}\n")
    result = solve(solver, query_id=args.id, query=args.query, context=context)

    output = json.dumps([result], ensure_ascii=False, indent=2)
    print("[Output JSON]")
    print(output)
