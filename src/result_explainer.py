"""
result_explainer.py - Module sinh Lời giải thích (Explanation) chuẩn sư phạm cho BTC ViPRA 2026.
Sử dụng: deepseek-ai/deepseek-math-7b-instruct (Dùng chung cho cả 2 Stage).
Tận dụng: Đề bài tiếng Việt + Tri thức RAG từ sách Feynman + Đáp án từ Solver.
"""

import re
import json
import urllib.request
import urllib.error
import os
import dspy
import config


def clean_explanation_text(text: str) -> str:
    """
    Làm sạch lời giải thích sư phạm:
    - Loại bỏ ký tự tiếng Trung, tiếng Nhật, tiếng Hàn nếu LLM đa ngữ tự sinh thừa.
    - Loại bỏ các tiền tố như 'Explanation:', 'Giải thích:', 'Lời giải:'.
    - Chuẩn hóa khoảng trắng.
    """
    if not text:
        return ""
    # Cắt bỏ mọi ký tự hoặc câu suy nghĩ tiếng Trung/nước ngoài mà model sinh thừa ở cuối
    cleaned = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+.*$", "", text, flags=re.DOTALL).strip()
    # Xóa tiền tố thừa
    cleaned = re.sub(r"^(?:Explanation|Giải thích|Lời giải thích|Hướng dẫn giải)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned



# ============================================================
# DSPY SIGNATURE & MODULE DÀNH CHO EXPLAINER (STAGE 2)
# ============================================================
class PhysicsExplainerSignature(dspy.Signature):
    """
    Bạn là Chuyên gia Vật lý và Giáo viên sư phạm xuất sắc.
    Nhiệm vụ: Viết một đoạn giải thích lý thuyết ngắn gọn (từ 1 đến 3 câu) bằng TIẾNG VIỆT cho bài toán vật lý.
    Yêu cầu:
    1. Nêu rõ hiện tượng vật lý và tên định luật / nguyên lý áp dụng (tham khảo RAG context nếu có).
    2. Viết công thức đại số tổng quát dưới dạng mã LaTeX ($...$).
    3. Trình bày ngắn gọn, súc tích, tự nhiên theo văn phong sách giáo khoa.
    4. Tuyệt đối KHÔNG viết CoT 4 bước dài dòng, KHÔNG chào hỏi, KHÔNG có từ thừa.
    """
    question: str = dspy.InputField(desc="Đề bài câu hỏi vật lý tiếng Việt.")
    context: str = dspy.InputField(desc="Lý thuyết và công thức liên quan từ RAG.")
    answer: str = dspy.InputField(desc="Đáp án cuối cùng đã tính toán.")
    unit: str = dspy.InputField(desc="Đơn vị vật lý chuẩn ASCII.")
    explanation: str = dspy.OutputField(
        desc="Đoạn văn ngắn (1-3 câu) giải thích hiện tượng, nêu định luật và công thức LaTeX ($...$)."
    )


class PhysicsExplainer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(PhysicsExplainerSignature)

    def forward(self, question: str, context: str = None, answer: str = "", unit: str = ""):
        if not context:
            import config
            context = getattr(
                config,
                "RAG_FALLBACK_TEXT",
                "Không có tài liệu tham khảo bổ sung. Hãy sử dụng các định luật vật lý tiêu chuẩn."
            )
        return self.predictor(
            question=question,
            context=context,
            answer=str(answer or ""),
            unit=str(unit or ""),
        )


_explainer_module = None

def get_optimized_explainer() -> PhysicsExplainer:
    """Nạp module PhysicsExplainer kèm checkpoint đã tối ưu (nếu có)."""
    global _explainer_module
    if _explainer_module is None:
        explainer = PhysicsExplainer()
        ckpt_path = getattr(config, "OPTIMIZED_EXPLAINER_PATH", "")
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                explainer.load(ckpt_path)
                print(f"[Explainer] Đã nạp thành công Optimized Explainer từ: {ckpt_path}")
            except Exception as e:
                print(f"[Explainer Cảnh báo] Không thể nạp checkpoint ({e}), dùng Signature mặc định.")
        _explainer_module = explainer
    return _explainer_module


# ==========================================
# PROMPT TEMPLATE DÀNH CHO GIÁO VIÊN VẬT LÝ SƯ PHẠM (DeepSeek-Math-7B)
# ==========================================

# Template Tiếng Anh: DeepSeek sinh lời giải thích lý thuyết thuần túy & công thức LaTeX (không số)
EXPLAINER_SYSTEM_PROMPT_EN = (
    "You are an expert physics educator. Your task is to provide a concise, rigorous, "
    "and pedagogical explanation of the physical phenomena, governing physical laws, "
    "and algebraic formulas in LaTeX ($...$). "
    "Never perform arithmetic calculations or substitute numbers into formulas."
)

EXPLAINER_USER_TEMPLATE_EN = """Write a concise physical explanation (2 to 3 sentences) in English for the following physics problem.

[PROBLEM]:
{question}

[THEORY CONTEXT]:
{context}

[TARGET RESULT]:
Answer: {answer}{unit}

[MANDATORY REQUIREMENTS]:
1. State the core physical phenomenon and explicitly identify the governing physical law, theorem, or principle applied.
2. Provide the general algebraic formula using LaTeX format ($...$).
3. STRICTLY FORBIDDEN: Do NOT perform arithmetic calculations or substitute numerical values into the formula. Focus purely on physical theory and algebraic relationships leading to the target result.
4. Keep the explanation concise and direct (2-3 sentences). No greeting, no conversational fillers.

Concise Physical Explanation:"""

# Template Tiếng Việt dự phòng (dùng khi tắt translation hoặc chạy trực tiếp)
EXPLAINER_SYSTEM_PROMPT = (
    "Bạn là Chuyên gia Vật lý và Giáo viên bộ môn Vật lý xuất sắc. "
    "Nhiệm vụ của bạn là giải thích bản chất vật lý, hiện tượng và công thức áp dụng "
    "của bài toán một cách ngắn gọn, chính xác và chuẩn mực sư phạm."
)

EXPLAINER_USER_TEMPLATE = """Hãy viết một đoạn giải thích lý thuyết ngắn gọn (khoảng 2 đến 4 câu) bằng TIẾNG VIỆT cho bài toán vật lý sau.

[ĐỀ BÀI]:
{question}

[TÀI LIỆU LÝ THUYẾT THAM KHẢO TỪ RAG]:
{context}

[KẾT QUẢ ĐÃ TÍNH TOÁN]:
Đáp số: {answer} {unit}

[YÊU CẦU BẮT BUỘC]:
1. Trình bày CƠ SỞ LÝ THUYẾT, bản chất vật lý và tên định luật/nguyên lý áp dụng (tham khảo tài liệu RAG nếu phù hợp).
2. Viết công thức ĐẠI SỐ TỔNG QUÁT dưới dạng mã LaTeX ($...$).
3. TUYỆT ĐỐI KHÔNG tự thực hiện phép tính số học hay thế số trung gian (để tránh sai lệch với đáp số đã xác định). Nêu rõ mối liên hệ giữa các đại lượng dẫn đến kết quả {answer} {unit}.
4. Đi thẳng vào giải thích lý thuyết, KHÔNG chào hỏi, KHÔNG có từ thừa như "Dưới đây là...".

Đoạn giải thích:"""


# ============================================================
# CLIENT GỌI LLM (vLLM hoặc Transformers)
# ============================================================
_local_pipe = None


def _call_vllm_api(prompt_messages: list, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Gọi endpoint chat completions của vLLM theo chuẩn OpenAI API."""
    url = f"{config.EXPLAINER_API_BASE}/chat/completions"
    payload = {
        "model": config.EXPLAINER_MODEL_ID,
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.EXPLAINER_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        if resp.status == 200:
            res_json = json.loads(resp.read().decode("utf-8"))
            return res_json["choices"][0]["message"]["content"].strip()
    return ""


def _call_local_transformers(prompt_messages: list, max_tokens: int = 512) -> str:
    """Nạp DeepSeek-Math-7B qua Hugging Face Pipeline nếu không có server vLLM."""
    global _local_pipe
    if _local_pipe is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        print(f"[Explainer] Đang nạp model {config.EXPLAINER_MODEL_ID} lên GPU...")
        tokenizer = AutoTokenizer.from_pretrained(config.EXPLAINER_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.EXPLAINER_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        _local_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt_text = _local_pipe.tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    out = _local_pipe(prompt_text, max_new_tokens=max_tokens, do_sample=False, temperature=0.1)
    gen_text = out[0]["generated_text"]
    if gen_text.startswith(prompt_text):
        return gen_text[len(prompt_text):].strip()
    return gen_text.strip()


def generate_explanation(
    question: str,
    context: str = None,
    answer: str = "",
    unit: str = "",
    question_en: str = None,
) -> str:
    """
    Hàm chính tạo trường 'explanation' chuẩn BTC:
    - DeepSeek-Math-7B sinh lời giải thích bằng TIẾNG ANH (tối ưu tư duy toán/vật lý & công thức LaTeX, tuyệt đối không tính số).
    - Dùng vinai-translate-en2vi-v2 dịch lời giải thích sang TIẾNG VIỆT trước khi nộp BTC.
    - Bảo toàn 100% công thức LaTeX ($...$) qua cơ chế Placeholder Protection bằng Python.
    - Có Fallback an toàn (không bao giờ để trống trường explanation).
    """
    if not config.USE_EXPLAINER:
        return _fallback_explanation(answer, unit)

    if not context or context == getattr(config, "RAG_FALLBACK_TEXT", ""):
        context_str = "Standard physical laws and principles in electromagnetism."
    else:
        # Cắt ngắn context RAG nếu quá dài để tiết kiệm context window
        context_str = context[:1500]

    unit_display = f" {unit}" if unit else ""

    # ============================================================
    # ƯU TIÊN 1: DSPy Optimized Explainer (Đã tối ưu hóa qua MIPROv2)
    # ============================================================
    ckpt_path = getattr(config, "OPTIMIZED_EXPLAINER_PATH", "")
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            # Tự động nạp LM cho DSPy nếu chưa có
            if getattr(dspy.settings, "lm", None) is None:
                try:
                    lm = dspy.LM(
                        model=f"openai/{config.BASE_MODEL_ID}",
                        api_base=config.LM_API_BASE,
                        api_key=config.LM_API_KEY,
                        max_tokens=512,
                        temperature=0.2,
                    )
                    dspy.settings.configure(lm=lm)
                except Exception:
                    pass

            if getattr(dspy.settings, "lm", None) is not None:
                explainer_mod = get_optimized_explainer()
                pred = explainer_mod(
                    question=question.strip(),
                    context=context_str.strip(),
                    answer=str(answer).strip(),
                    unit=str(unit).strip(),
                )
                opt_exp = clean_explanation_text(str(getattr(pred, "explanation", "")))
                if opt_exp and len(opt_exp) >= 20:
                    return opt_exp
        except Exception:
            pass

    # ============================================================
    # ƯU TIÊN 2: LLM sinh Tiếng Anh → VinAI dịch sang Tiếng Việt
    # ============================================================
    if getattr(config, "USE_TRANSLATION", True):
        try:
            from result_optimize_translator import translate_en_to_vi, translate_vi_to_en

            # Ưu tiên lấy câu hỏi tiếng Anh đã có sẵn từ Solver (tránh dịch lại lần 2)
            q_en = question_en.strip() if question_en else translate_vi_to_en(question)

            user_prompt_en = EXPLAINER_USER_TEMPLATE_EN.format(
                question=q_en,
                context=context_str.strip(),
                answer=str(answer).strip(),
                unit=unit_display,
            )
            messages_en = [
                {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT_EN},
                {"role": "user", "content": user_prompt_en},
            ]

            # Bước A: LLM sinh lời giải thích tiếng Anh (vLLM hoặc Transformers)
            exp_en = ""
            try:
                exp_en = _call_vllm_api(messages_en)
            except Exception:
                try:
                    exp_en = _call_local_transformers(messages_en)
                except Exception:
                    exp_en = ""

            # Bước B: Dịch lời giải thích tiếng Anh sang tiếng Việt qua VinAI (có bảo vệ LaTeX bằng Python)
            if exp_en and len(exp_en) >= 20:
                exp_vi = clean_explanation_text(translate_en_to_vi(exp_en))
                if exp_vi and len(exp_vi) >= 15:
                    return exp_vi
        except Exception:
            pass

    # ============================================================
    # ƯU TIÊN 3: Sinh Tiếng Việt trực tiếp (nếu tắt translation hoặc lỗi dịch)
    # ============================================================
    user_prompt_vi = EXPLAINER_USER_TEMPLATE.format(
        question=question.strip(),
        context=context_str.strip(),
        answer=str(answer).strip(),
        unit=unit_display,
    )
    messages_vi = [
        {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_vi},
    ]

    # Ưu tiên dự phòng 1: Gọi vLLM trực tiếp
    try:
        explanation = clean_explanation_text(_call_vllm_api(messages_vi))
        if explanation and len(explanation) >= 20:
            return explanation
    except Exception:
        pass

    # Ưu tiên dự phòng 2: Gọi Transformers Local
    try:
        explanation = clean_explanation_text(_call_local_transformers(messages_vi))
        if explanation and len(explanation) >= 20:
            return explanation
    except Exception:
        pass

    # ============================================================
    # ƯU TIÊN 4: Fallback Programmatic (Bảo hiểm 100% không để trống trường explanation)
    # ============================================================
    return clean_explanation_text(_fallback_explanation(answer, unit))


def _fallback_explanation(answer: str, unit: str) -> str:
    """Fallback an toàn khi chưa nạp được LLM Explainer."""
    unit_str = f" {unit}" if unit else ""
    if answer and answer != "ERROR":
        return f"Áp dụng các định luật và công thức vật lý tổng quát trong Điện và Từ học để tính toán kết quả: final_answer = {answer}{unit_str}."
    return "Áp dụng các định luật vật lý tiêu chuẩn để suy luận và giải bài toán."
