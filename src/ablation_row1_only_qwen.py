"""
ablation_row1_only_qwen.py - Cấu hình Ablation Row 1: Only Qwen (Direct CoT Baseline)

Đặc điểm cấu hình Row 1:
- KHÔNG dùng Hybrid RAG (tự nhớ công thức vật lý từ bộ nhớ tham số).
- KHÔNG dùng Translation (đọc trực tiếp đề bài tiếng Việt).
- KHÔNG dùng PAL & Sandbox (tự tính nhẩm số học trong quá trình sinh văn bản).
- KHÔNG tách rời Explainer (tự sinh đồng thời answer, unit, explanation, reasoning.steps trong 1 lượt gọi duy nhất).
- Đầu ra: Vẫn chuẩn hóa đúng cấu trúc JSON 5 trường của BTC để so sánh công bằng với Row 2, 3, 4.
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
from result_explainer import clean_explanation_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Ablation_Row1_OnlyQwen")


# ============================================================
# 1. PLACEHOLDER PROMPT / SIGNATURE CHO ROW 1 (SẼ TINH CHỈNH SAU)
# ============================================================

class Row1DirectCoTSignature(dspy.Signature):
    """
    [TODO: PROMPT ROW 1 SẼ ĐƯỢC CẬP NHẬT TẠI ĐÂY]
    Giải bài toán vật lý bằng cách suy luận từng bước (Chain-of-Thought),
    tự tính toán kết quả cuối cùng, đơn vị chuẩn ASCII và viết lời giải thích lý thuyết.
    """
    question: str = dspy.InputField(
        desc="Đề bài vật lý bằng tiếng Việt."
    )
    reasoning_steps: str = dspy.OutputField(
        desc="Các bước suy luận và tính toán từng bước (mỗi bước 1 dòng)."
    )
    explanation: str = dspy.OutputField(
        desc="Lời giải thích lý thuyết và công thức vật lý bằng tiếng Việt."
    )
    answer: str = dspy.OutputField(
        desc="Kết quả số học cuối cùng (hoặc kết luận ngắn gọn). Chỉ ghi giá trị, không kèm đơn vị."
    )
    unit: str = dspy.OutputField(
        desc="Đơn vị vật lý chuẩn ASCII (VD: V, A, ohm, J, Hz). Để rỗng nếu không có đơn vị."
    )


class Row1DirectCoTSolver(dspy.Module):
    """Module Row 1: Qwen2.5-7B tự gánh toàn bộ đọc hiểu, nhớ công thức, tính nhẩm và giải thích trong 1 lượt."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(Row1DirectCoTSignature)

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


def _format_unit(unit_val) -> str:
    if isinstance(unit_val, (list, tuple)):
        clean_parts = [str(u).strip() for u in unit_val if str(u).strip() and str(u).strip() != "-"]
        return "; ".join(clean_parts)
    u_str = str(unit_val).strip() if unit_val is not None else ""
    return "" if u_str == "-" else u_str


def _parse_steps(raw_steps: str) -> List[str]:
    if not raw_steps:
        return []
    return [line.strip() for line in str(raw_steps).splitlines() if line.strip()]


# ============================================================
# 3. KHỞI TẠO MÔ HÌNH & HÀM GIẢI CHÍNH (ROW 1)
# ============================================================

row1_solver_instance: Optional[Row1DirectCoTSolver] = None


def init_row1_solver() -> Optional[Row1DirectCoTSolver]:
    """Khởi tạo kết nối vLLM / Transformers cho Row 1."""
    try:
        from optimize_prompt import setup_lm
        setup_lm()
        solver = Row1DirectCoTSolver()
        logger.info("[Row 1] Khởi tạo Row1DirectCoTSolver thành công!")
        return solver
    except Exception as e:
        logger.warning(f"[Row 1] Không thể khởi tạo LM ({e}).")
        return None


def solve_row1(query_id: str, query: str) -> dict:
    """
    Thực thi pipeline Row 1 (Only Qwen - Direct CoT) cho 1 câu hỏi:
    - Nhận trực tiếp câu hỏi tiếng Việt.
    - Gọi 1 lượt duy nhất để lấy answer, unit, explanation, reasoning_steps.
    """
    global row1_solver_instance
    if row1_solver_instance is None:
        row1_solver_instance = init_row1_solver()

    try:
        pred = row1_solver_instance(question=query)
        raw_ans = str(getattr(pred, "answer", "")).strip()
        raw_unit = _format_unit(getattr(pred, "unit", ""))
        raw_exp = clean_explanation_text(str(getattr(pred, "explanation", "")))
        raw_steps = str(getattr(pred, "reasoning_steps", "")).strip()

        if not raw_exp:
            raw_exp = "Áp dụng định luật vật lý tiêu chuẩn để tính toán kết quả."
        steps = _parse_steps(raw_steps)
        if not steps:
            steps = [raw_exp]

        return {
            "query_id": query_id,
            "answer": raw_ans if raw_ans else "0",
            "unit": raw_unit,
            "explanation": raw_exp,
            "reasoning": {
                "type": "cot",
                "steps": steps,
            },
        }
    except Exception as e:
        logger.error(f"[Row 1] Lỗi khi xử lý [{query_id}]: {e}", exc_info=True)
        return {
            "query_id": query_id,
            "answer": "0",
            "unit": "",
            "explanation": "Áp dụng các định luật vật lý tiêu chuẩn để suy luận bài toán.",
            "reasoning": {
                "type": "cot",
                "steps": ["[Lỗi sinh phản hồi trực tiếp ở Row 1]"],
            },
        }


# ============================================================
# 4. FASTAPI ENDPOINT (TƯƠNG THÍCH 100% VỚI SERVER.PY)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global row1_solver_instance
    logger.info("=== KHỞI ĐỘNG SERVER ABLATION ROW 1 (ONLY QWEN - DIRECT COT) ===")
    row1_solver_instance = init_row1_solver()
    yield


app = FastAPI(title="ViPRA 2026 - Ablation Row 1 (Only Qwen)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=List[PredictOutputItem])
async def predict_endpoint(payload: PredictInput):
    res = solve_row1(payload.query_id.strip(), payload.query.strip())
    return [PredictOutputItem(**res)]
