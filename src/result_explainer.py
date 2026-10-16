"""
result_explainer.py - Module sinh Lời giải thích (Explanation) chuẩn sư phạm cho BTC ViPRA 2026.
Sử dụng: Qwen/Qwen2.5-7B-Instruct (Stage 2 Explainer).
Tận dụng: Đề bài tiếng Việt + Tri thức RAG từ sách Feynman + Đáp án từ Solver.
"""

import json
import urllib.request
import urllib.error
import os
import dspy
import config


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

    def forward(self, question: str, context: str, answer: str, unit: str):
        return self.predictor(question=question, context=context, answer=answer, unit=unit)


# ============================================================
# PROMPT TEMPLATE DÀNH CHO GIÁO VIÊN VẬT LÝ SƯ PHẠM (Qwen2.5-7B)
# ============================================================
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
    """Nạp Qwen2.5-7B qua Hugging Face Pipeline nếu không có server vLLM."""
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
) -> str:
    """
    Hàm chính tạo trường 'explanation' chuẩn BTC:
    - Nhận câu hỏi tiếng Việt, context từ RAG, và đáp án/đơn vị từ Solver.
    - Sinh đoạn văn lý thuyết tiếng Việt 2-4 câu chuẩn LaTeX.
    - Có cơ chế Fallback an toàn (không bao giờ để trống theo luật BTC).
    """
    if not config.USE_EXPLAINER:
        return _fallback_explanation(answer, unit)

    if not context or context == getattr(config, "RAG_FALLBACK_TEXT", ""):
        context_str = "Sử dụng các định luật và công thức vật lý chuẩn trong chương trình Điện và Từ học."
    else:
        # Cắt ngắn context RAG nếu quá dài để tiết kiệm context window
        context_str = context[:1500]

    unit_display = f" {unit}" if unit else ""
    user_prompt = EXPLAINER_USER_TEMPLATE.format(
        question=question.strip(),
        context=context_str.strip(),
        answer=str(answer).strip(),
        unit=unit_display,
    )

    messages = [
        {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Ưu tiên 1: Gọi vLLM API
    try:
        explanation = _call_vllm_api(messages)
        if explanation:
            return explanation
    except Exception:
        pass

    # Ưu tiên 2: Gọi Transformers Local nếu không có vLLM
    try:
        explanation = _call_local_transformers(messages)
        if explanation:
            return explanation
    except Exception:
        pass

    # Ưu tiên 3: Fallback Programmatic (Bảo hiểm 100% không để trống trường explanation)
    return _fallback_explanation(answer, unit)


def _fallback_explanation(answer: str, unit: str) -> str:
    """Fallback an toàn khi chưa nạp được LLM Explainer."""
    unit_str = f" {unit}" if unit else ""
    if answer and answer != "ERROR":
        return f"Áp dụng các định luật và công thức vật lý tổng quát trong Điện và Từ học để tính toán kết quả: final_answer = {answer}{unit_str}."
    return "Áp dụng các định luật vật lý tiêu chuẩn để suy luận và giải bài toán."
