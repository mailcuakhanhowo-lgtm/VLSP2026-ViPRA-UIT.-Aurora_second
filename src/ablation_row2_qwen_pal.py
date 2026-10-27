"""
ablation_row2_qwen_pal.py - Cấu hình Ablation Row 2: Qwen + PAL (Mục 2.3, không có RAG & chưa tách Explainer)

Đặc điểm cấu hình Row 2:
- CÓ dùng Symbol-Masking Translation (dịch đề bài sang tiếng Anh Q_en bảo toàn ký hiệu).
- CÓ dùng PAL & Sandbox (Qwen2.5-7B sinh mã Python -> thực thi trong Sandbox lấy final_answer và unit chuẩn).
- KHÔNG dùng Hybrid RAG (không có ngữ cảnh lý thuyết Feynman K, mô hình tự nhớ công thức).
- KHÔNG tách rời Explainer (Qwen phải vừa sinh python_code vừa tự viết trường explanation trong cùng 1 lượt sinh).
- Đầu ra: Chuẩn hóa đúng cấu trúc JSON 5 trường của BTC.
"""

import os
import sys
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Ablation_Row2_QwenPAL")


# ============================================================
# 1. PLACEHOLDER PROMPT / SIGNATURE CHO ROW 2 (SẼ TINH CHỈNH SAU)
# ============================================================

class Row2JointPALSignature(dspy.Signature):
    """
    [TODO: PROMPT ROW 2 SẼ ĐƯỢC CẬP NHẬT TẠI ĐÂY]
    Đọc đề bài vật lý (không có tài liệu RAG tham khảo), tự xác định định luật vật lý:
    1. Viết lời giải thích lý thuyết bằng tiếng Việt vào trường 'explanation'.
    2. Viết mã Python thực thi (3 phần: khai báo SI, chú thích công thức, gán final_answer và unit) vào trường 'python_code'.
    """
    question: str = dspy.InputField(
        desc="Physics question to solve (translated with preserved symbols)."
    )
    explanation: str = dspy.OutputField(
        desc="Vietnamese theoretical explanation of the physical principles and formulas."
    )
    python_code: str = dspy.OutputField(
        desc="Executable Python code defining final_answer and unit. No markdown fences."
    )


class Row2JointPALSolver(dspy.Module):
    """Module Row 2: Sinh đồng thời explanation + python_code trong 1 lượt mà KHÔNG có RAG context."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(Row2JointPALSignature)

    def forward(self, question: str):
        return self.predictor(question=question)


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
    return "" if u_str == "-" else u_str


def _build_reasoning_steps(clean_code: str) -> List[str]:
    steps = []
    for line in clean_code.splitlines():
        line = line.strip()
        if line and not line.startswith(("import ", "from ")):
            steps.append(line)
    return steps


# ============================================================
# 3. KHỞI TẠO MÔ HÌNH & HÀM GIẢI CHÍNH (ROW 2)
# ============================================================

row2_solver_instance: Optional[Row2JointPALSolver] = None


def init_row2_solver() -> Optional[Row2JointPALSolver]:
    """Khởi tạo kết nối vLLM / Transformers cho Row 2."""
    try:
        from optimize_prompt import setup_lm
        setup_lm()
        solver = Row2JointPALSolver()
        logger.info("[Row 2] Khởi tạo Row2JointPALSolver thành công!")
        return solver
    except Exception as e:
        logger.warning(f"[Row 2] Không thể khởi tạo LM ({e}).")
        return None


def solve_row2(query_id: str, query: str) -> dict:
    """
    Thực thi pipeline Row 2 (Qwen + PAL, không có RAG, chưa tách Explainer):
    1. Dịch câu hỏi Vi -> En bảo toàn ký hiệu (Q_en).
    2. Qwen2.5-7B sinh đồng thời 'explanation' và 'python_code' trong 1 lượt (không có RAG).
    3. Thực thi 'python_code' trong Sandbox để lấy final_answer và unit.
    4. Đóng gói JSON 5 trường chuẩn BTC.
    """
    global row2_solver_instance
    if row2_solver_instance is None:
        row2_solver_instance = init_row2_solver()

    try:
        # Bước 1: Dịch bảo toàn ký hiệu (Mục 2.3)
        if getattr(config, "USE_TRANSLATION", True):
            question_for_llm = translate_vi_to_en(query)
        else:
            question_for_llm = query

        # Bước 2: Gọi Qwen sinh chung cả explanation và python_code (KHÔNG có RAG context)
        pred = row2_solver_instance(question=question_for_llm)
        raw_code = str(getattr(pred, "python_code", "")).strip()
        raw_exp = clean_explanation_text(str(getattr(pred, "explanation", "")))

        # Bước 3: Chạy Sandbox exec()
        clean_code = clean_python_code(raw_code)
        final_answer, unit = execute_code(clean_code)

        if final_answer is None:
            answer_str = "0"
            unit_str = ""
            steps = ["[Lỗi thực thi mã Python ở Row 2]", raw_code] if raw_code else ["[Không sinh được mã thực thi]"]
        else:
            answer_str = _format_final_answer(final_answer)
            unit_str = _format_unit(unit)
            steps = _build_reasoning_steps(clean_code)

        if not raw_exp:
            raw_exp = "Áp dụng các định luật vật lý tiêu chuẩn để suy luận và giải bài toán."

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
        logger.error(f"[Row 2] Lỗi khi xử lý [{query_id}]: {e}", exc_info=True)
        return {
            "query_id": query_id,
            "answer": "0",
            "unit": "",
            "explanation": "Áp dụng các định luật vật lý tiêu chuẩn để suy luận bài toán.",
            "reasoning": {
                "type": "cot",
                "steps": ["[Lỗi xử lý ngoại lệ ở Row 2]"],
            },
        }


# ============================================================
# 4. FASTAPI ENDPOINT (TƯƠNG THÍCH 100% VỚI SERVER.PY)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global row2_solver_instance
    logger.info("=== KHỞI ĐỘNG SERVER ABLATION ROW 2 (QWEN + PAL, NO RAG, NO DECOUPLED EXPLAINER) ===")
    row2_solver_instance = init_row2_solver()
    if getattr(config, "USE_TRANSLATION", True):
        try:
            preload_translator(direction="vi2en")
        except Exception as te:
            logger.warning(f"[Row 2] Không thể preload translator: {te}")
    yield


app = FastAPI(title="ViPRA 2026 - Ablation Row 2 (Qwen + PAL)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=List[PredictOutputItem])
async def predict_endpoint(payload: PredictInput):
    res = solve_row2(payload.query_id.strip(), payload.query.strip())
    return [PredictOutputItem(**res)]
