"""
ablation_row3_qwen_rag_pal.py - Cấu hình Ablation Row 3: Qwen + RAG + PAL (Mục 2.2 + 2.3, chưa tách Explainer)

Đặc điểm cấu hình Row 3:
- CÓ dùng Hybrid RAG (trích xuất lý thuyết K từ 1,005 chunks giáo trình Feynman).
- CÓ dùng Symbol-Masking Translation (dịch đề bài sang tiếng Anh Q_en bảo toàn ký hiệu).
- CÓ dùng PAL & Sandbox (Qwen2.5-7B sinh mã Python -> thực thi trong Sandbox lấy final_answer và unit chuẩn).
- KHÔNG tách rời Explainer (Qwen nhận K và Q_en, nhưng phải vừa sinh python_code vừa tự viết trường explanation bằng TIẾNG VIỆT trong cùng 1 lượt sinh, chưa dùng result_explainer.py tách rời).
- Đầu ra: Chuẩn hóa đúng cấu trúc JSON 5 trường của BTC.
"""

import os
import sys
import json
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import dspy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from result_optimize_metric import clean_python_code, execute_code
from result_optimize_translator import translate_vi_to_en, preload as preload_translator
from result_explainer import clean_explanation_text
from ablation_row2_qwen_pal import DEFAULT_DEMO_EXPLANATIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Ablation_Row3_QwenRAGPAL")


# ============================================================
# 1. PROMPT & 4 FEW-SHOT DEMOS CHO ROW 3 (KẾ THỪA TỪ OPTIMIZED PROMPTS)
# ============================================================

class Row3RAGJointPALSignature(dspy.Signature):
    """
    You are an expert in Physics, physics education, and Python programming following the Program-Aided Language (PAL) approach.
    Read the physics question and the provided RAG context (relevant electromagnetism theory and formulas from the Feynman Lectures).
    You must perform two tasks simultaneously in a single pass:
    (A) Write a concise theoretical explanation in VIETNAMESE into 'explanation'.
    (B) Generate clean, executable Python code into 'python_code' to solve the problem numerically.

    RAG Context Grounding & Fallback Rules:
    - Cross-reference and apply the physical laws and algebraic formulas provided in 'context' to both explain the phenomenon in 'explanation' and program the expression in 'python_code'.
    - Fallback: If 'context' does not directly contain the specific formula needed for the problem, rely on standard physical laws and formulas in Electromagnetism.

    Instructions for 'explanation' (MANDATORY VIETNAMESE - Criterion P2):
    1. Although the input question and context are in English, 'explanation' MUST BE WRITTEN IN VIETNAMESE (1 to 3 concise sentences).
    2. State the core physical phenomenon and the governing physical law/principle applied (referencing 'context' when applicable).
    3. Write the general algebraic formula in LaTeX format ($...$).
    4. Do NOT perform intermediate arithmetic calculations in 'explanation', do NOT write verbose 4-step CoT, and do NOT include greetings.

    Instructions for 'python_code' (Criterion P1 & P3):
    1. Structure your code into 3 parts:
       - Part 1: Declare given variables converted to standard SI units.
       - Part 2: Comment the algebraic physics formula used.
       - Part 3: Calculate the result and assign to 'final_answer', and assign the standard ASCII unit to 'unit'.
    2. For multi-part questions, assign a list of answers to 'final_answer' (e.g., final_answer = [ans1, ans2]).
    3. For qualitative/theoretical questions without numerical values, assign the concise conclusion string to 'final_answer' and '' to 'unit'.
    4. Trigonometry with degrees: Use pre-injected functions directly: cos_deg(deg), sin_deg(deg), tan_deg(deg). DO NOT wrap math.radians.
    5. Available physical constants: k = 9e9, eps0 = 8.854e-12, mu0 = 4*math.pi*1e-7, g = 9.8, pi = math.pi.
    6. DO NOT wrap code in markdown fences (```python). Return only raw executable Python code.
    """
    context: str = dspy.InputField(
        desc="Physics theory, electromagnetism laws, and formulas retrieved from Feynman RAG."
    )
    question: str = dspy.InputField(
        desc="Physics question to solve (translated into English with preserved symbols)."
    )
    explanation: str = dspy.OutputField(
        desc="Đoạn văn ngắn (1-3 câu) BẮT BUỘC BẰNG TIẾNG VIỆT giải thích hiện tượng, nêu định luật và công thức LaTeX ($...$)."
    )
    python_code: str = dspy.OutputField(
        desc="Clean executable Python code defining final_answer and unit. No markdown fences."
    )


def _load_row3_demos() -> List[dspy.Example]:
    """Tải 2 demos chuẩn từ optimized_prompt.json (cắt gọn context để tiết kiệm VRAM, gắn thêm explanation tiếng Việt)."""
    demos = []
    if os.path.exists(config.OPTIMIZED_PROMPT_PATH):
        try:
            with open(config.OPTIMIZED_PROMPT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_demos = data.get("predictor", {}).get("demos", [])
            for idx, d in enumerate(raw_demos[:2]):
                exp_vi = DEFAULT_DEMO_EXPLANATIONS[idx] if idx < len(DEFAULT_DEMO_EXPLANATIONS) else DEFAULT_DEMO_EXPLANATIONS[0]
                raw_ctx = str(d.get("context", config.RAG_FALLBACK_TEXT)).strip()
                concise_ctx = raw_ctx[:350].strip() + "..." if len(raw_ctx) > 350 else raw_ctx
                demos.append(
                    dspy.Example(
                        context=concise_ctx,
                        question=d.get("question", ""),
                        explanation=exp_vi,
                        python_code=d.get("python_code", ""),
                    ).with_inputs("context", "question")
                )
        except Exception as e:
            logger.warning(f"[Row 3] Không thể đọc demos từ {config.OPTIMIZED_PROMPT_PATH}: {e}")
    return demos


class Row3RAGJointPALSolver(dspy.Module):
    """Module Row 3: Có RAG context K, nhưng vẫn bắt Qwen sinh chung explanation + python_code trong 1 lượt."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(Row3RAGJointPALSignature)
        demos = _load_row3_demos()
        if demos:
            self.predictor.demos = demos

    def forward(self, question: str, context: str = None):
        if not context:
            context = getattr(
                config,
                "RAG_FALLBACK_TEXT",
                "Không có tài liệu tham khảo bổ sung. Hãy sử dụng các định luật vật lý tiêu chuẩn."
            )
        return self.predictor(context=context, question=question)


# ============================================================
# 2. PYDANTIC SCHEMAS & HELPER FORMATTING (CHUẨN 5 TRƯỜNG BTC)
# ============================================================

class PredictInput(BaseModel):
    query_id: str = Field(..., description="Định danh truy vấn duy nhất.")
    query: str = Field(..., description="Toàn bộ bài toán bằng tiếng Việt.")


class ReasoningOutput(BaseModel):
    type: str = Field(default="cot")
    steps: List[str] = Field(default_factory=list)


class PredictOutputItem(BaseModel):
    query_id: str
    answer: str
    unit: str = Field(default="")
    explanation: str
    reasoning: Optional[ReasoningOutput] = Field(default=None)


def _format_number(val: float) -> str:
    if val == 0:
        return "0"
    abs_val = abs(val)
    if abs_val < 1e-4 or abs_val >= 1e6:
        formatted = f"{val:.4e}"
        mantissa, exp = formatted.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp_int = int(exp)
        return f"{mantissa} * 10^{exp_int}"
    else:
        return f"{val:.6f}".rstrip("0").rstrip(".")


def _format_final_answer(final_answer) -> str:
    if isinstance(final_answer, (list, tuple)):
        parts = []
        for v in final_answer:
            try:
                parts.append(_format_number(float(v)))
            except (ValueError, TypeError):
                parts.append(str(v).strip())
        return "; ".join(parts)
    try:
        return _format_number(float(final_answer))
    except (ValueError, TypeError):
        return str(final_answer).strip()


def _format_unit(unit_val) -> str:
    if isinstance(unit_val, (list, tuple)):
        clean_parts = [str(u).strip() for u in unit_val if str(u).strip() and str(u).strip() != "-"]
        return "; ".join(clean_parts)
    u_str = str(unit_val).strip() if unit_val is not None else ""
    return "" if u_str in ("-", "''", '""', "None") else u_str


def _build_reasoning_steps(clean_code: str) -> List[str]:
    steps = []
    for line in clean_code.splitlines():
        line = line.strip()
        if line and not line.startswith(("import ", "from ")):
            steps.append(line)
    return steps


# ============================================================
# 3. KHỞI TẠO MÔ HÌNH, RAG & HÀM GIẢI CHÍNH (ROW 3)
# ============================================================

row3_solver_instance: Optional[Row3RAGJointPALSolver] = None
row3_rag_instance = None


def init_row3_rag():
    """Khởi tạo RAG Pipeline (1,005 Feynman chunks)."""
    global row3_rag_instance
    try:
        from rag_module import RAGPipeline
        rag = RAGPipeline()
        rag.load_indices()
        logger.info("[Row 3] Nạp RAG Indices (BM25 + FAISS BGE-M3) thành công!")
        return rag
    except Exception as e:
        logger.warning(f"[Row 3] Không thể nạp RAG DB ({e}). Dùng fallback context.")
        return None


def init_row3_solver() -> Optional[Row3RAGJointPALSolver]:
    """Khởi tạo kết nối vLLM / Transformers cho Row 3."""
    try:
        if getattr(dspy.settings, "lm", None) is None:
            from optimize_prompt import setup_lm
            setup_lm()
        solver = Row3RAGJointPALSolver()
        logger.info(f"[Row 3] Khởi tạo Row3RAGJointPALSolver kèm {len(solver.predictor.demos)} demos thành công!")
        return solver
    except Exception as e:
        logger.warning(f"[Row 3] Không thể khởi tạo LM ({e}).")
        return None


def solve_row3(query_id: str, query: str, precomputed_context: str = None) -> dict:
    """
    Thực thi pipeline Row 3 (Qwen + RAG + PAL, chưa tách Explainer):
    1. Truy xuất lý thuyết Feynman K từ Hybrid RAG (Mục 2.2).
    2. Dịch câu hỏi Vi -> En bảo toàn ký hiệu Q_en (Mục 2.3).
    3. Qwen2.5-7B nhận cả K và Q_en, sinh đồng thời 'explanation' và 'python_code' trong 1 lượt.
    4. Thực thi 'python_code' trong Sandbox để lấy final_answer và unit.
    5. Đóng gói JSON 5 trường chuẩn BTC.
    """
    global row3_solver_instance, row3_rag_instance
    if row3_solver_instance is None:
        row3_solver_instance = init_row3_solver()

    try:
        # Bước 0: Truy xuất tri thức lý thuyết từ Hybrid RAG (Mục 2.2)
        if precomputed_context is not None:
            context = precomputed_context
        else:
            if row3_rag_instance is None:
                row3_rag_instance = init_row3_rag()
            context = config.RAG_FALLBACK_TEXT
            if row3_rag_instance:
                try:
                    context = row3_rag_instance.get_context(query)
                except Exception as re:
                    logger.warning(f"[Row 3] Lỗi truy xuất RAG cho [{query_id}]: {re}")

        # Giới hạn context tối đa ~1500 ký tự để an toàn VRAM tuyệt đối trên 2x T4
        if context and len(context) > 1500:
            context = context[:1500].rsplit("\n\n", 1)[0]

        # Bước 1: Dịch bảo toàn ký hiệu (Mục 2.3)
        if getattr(config, "USE_TRANSLATION", True):
            question_for_llm = translate_vi_to_en(query)
        else:
            question_for_llm = query

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Bước 2: Gọi Qwen sinh chung cả explanation và python_code (CÓ RAG context K)
        pred = row3_solver_instance(question=question_for_llm, context=context)
        raw_code = str(getattr(pred, "python_code", "")).strip()
        raw_exp = clean_explanation_text(str(getattr(pred, "explanation", "")))

        # Bước 3: Chạy Sandbox exec()
        clean_code = clean_python_code(raw_code)
        final_answer, unit = execute_code(clean_code)

        if final_answer is None:
            answer_str = "0"
            unit_str = ""
            steps = ["[Lỗi thực thi mã Python ở Row 3]", raw_code] if raw_code else ["[Không sinh được mã thực thi]"]
        else:
            answer_str = _format_final_answer(final_answer)
            unit_str = _format_unit(unit)
            steps = _build_reasoning_steps(clean_code)

        if not raw_exp:
            raw_exp = "Áp dụng các định luật vật lý tiêu chuẩn trong Điện và Từ học để suy luận và giải bài toán."

        return {
            "query_id": query_id,
            "answer": answer_str,
            "unit": unit_str,
            "explanation": raw_exp,
            "reasoning": {
                "type": "cot",
                "steps": steps,
            },
        }
    except Exception as e:
        logger.error(f"[Row 3] Lỗi khi xử lý [{query_id}]: {e}", exc_info=True)
        return {
            "query_id": query_id,
            "answer": "0",
            "unit": "",
            "explanation": "Áp dụng các định luật vật lý tiêu chuẩn để suy luận bài toán.",
            "reasoning": {
                "type": "cot",
                "steps": ["[Lỗi xử lý ngoại lệ ở Row 3]"],
            },
        }


# ============================================================
# 4. FASTAPI ENDPOINT (TƯƠNG THÍCH 100% VỚI SERVER.PY)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global row3_solver_instance, row3_rag_instance
    logger.info("=== KHỞI ĐỘNG SERVER ABLATION ROW 3 (QWEN + RAG + PAL, NO DECOUPLED EXPLAINER) ===")
    row3_solver_instance = init_row3_solver()
    row3_rag_instance = init_row3_rag()
    if getattr(config, "USE_TRANSLATION", True):
        try:
            preload_translator(direction="vi2en")
        except Exception as te:
            logger.warning(f"[Row 3] Không thể preload translator: {te}")
    yield


app = FastAPI(title="ViPRA 2026 - Ablation Row 3 (Qwen + RAG + PAL)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=List[PredictOutputItem])
async def predict_endpoint(payload: PredictInput):
    res = solve_row3(payload.query_id.strip(), payload.query.strip())
    return [PredictOutputItem(**res)]
