"""
ablation_row1_only_qwen.py - Cấu hình Ablation Row 1: Only Qwen (Direct CoT Baseline)

Đặc điểm cấu hình Row 1:
- KHÔNG dùng Hybrid RAG (tự nhớ công thức vật lý từ bộ nhớ tham số).
- KHÔNG dùng Translation (đọc trực tiếp đề bài tiếng Việt).
- KHÔNG dùng PAL & Sandbox (tự tính nhẩm số học trong quá trình sinh văn bản).
- KHÔNG tách rời Explainer (tự sinh đồng thời reasoning_steps, explanation, answer, unit trong 1 lượt gọi duy nhất).
- Đầu ra: Vẫn chuẩn hóa đúng cấu trúc JSON 5 trường của BTC để so sánh công bằng với Row 2, 3, 4.
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
from result_explainer import clean_explanation_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Ablation_Row1_OnlyQwen")


# ============================================================
# 1. PROMPT & 4 FEW-SHOT DEMOS CHO ROW 1 (KẾ THỪA TỪ OPTIMIZED PROMPTS)
# ============================================================

class Row1DirectCoTSignature(dspy.Signature):
    """
    Bạn là Chuyên gia Vật lý và Giáo viên sư phạm xuất sắc.
    Đọc kỹ đề bài vật lý bằng tiếng Việt và giải bài toán trực tiếp trong một lượt duy nhất (tự nhớ định luật, tự tính toán số học, không dùng code Python).

    1. Yêu cầu cho 'reasoning_steps' (Chuỗi lập luận CoT 4 bước - Tiêu chí P3):
       - Trình bày suy luận qua đúng 4 bước, mỗi bước là một câu hoàn chỉnh nằm trên một dòng riêng biệt:
         + Bước 1: Nhận diện các dữ kiện đề bài đã cho và quy đổi về đơn vị chuẩn hệ SI.
         + Bước 2: Xác định định luật vật lý và công thức đại số áp dụng (hằng số chuẩn: k = 9e9, eps0 = 8.854e-12, mu0 = 4*pi*1e-7, g = 9.8).
         + Bước 3: Thay số vào công thức và thực hiện phép tính số học cụ thể.
         + Bước 4: Kết luận giá trị đáp án cuối cùng và đơn vị đo.

    2. Yêu cầu cho 'explanation' (Giải thích lý thuyết - Tiêu chí P2):
       - Viết một đoạn văn ngắn gọn (từ 1 đến 3 câu) bằng TIẾNG VIỆT.
       - Nêu rõ hiện tượng vật lý, định luật áp dụng và viết công thức đại số tổng quát dưới dạng mã LaTeX ($...$).
       - Tuyệt đối KHÔNG nháp phép tính số học vào trường này, KHÔNG chào hỏi.

    3. Yêu cầu BẮT BUỘC cho 'answer' và 'unit' (Độ chính xác - Tiêu chí P1):
       - 'answer': CHỈ xuất duy nhất con số kết quả (ví dụ: 0.045, hoặc dạng khoa học a * 10^b nếu quá nhỏ/lớn), hoặc kết luận ngắn gọn đối với câu hỏi lý thuyết định tính. NGHIÊM CẤM viết kèm đơn vị vào trường 'answer' (ví dụ: viết '0.045', KHÔNG viết '0.045 J'). Với bài toán nhiều ý, phân cách các kết quả bằng '; ' (ví dụ: '1.5; 2.4').
       - 'unit': Bắt buộc xuất đơn vị chuẩn ASCII (ví dụ: J, V, A, ohm, Hz, uF, %). Nếu câu hỏi không có đơn vị hoặc là câu lý thuyết, để chuỗi rỗng ''.
    """
    question: str = dspy.InputField(
        desc="Đề bài vật lý bằng tiếng Việt."
    )
    reasoning_steps: str = dspy.OutputField(
        desc="4 bước suy luận CoT, mỗi bước trên 1 dòng riêng biệt (Bước 1 -> Bước 4)."
    )
    explanation: str = dspy.OutputField(
        desc="Đoạn văn ngắn (1-3 câu) bằng tiếng Việt giải thích hiện tượng và công thức LaTeX ($...$), không chứa phép tính số."
    )
    answer: str = dspy.OutputField(
        desc="Chỉ ghi duy nhất giá trị số kết quả (hoặc kết luận ngắn gọn). TUYỆT ĐỐI KHÔNG kèm đơn vị."
    )
    unit: str = dspy.OutputField(
        desc="Đơn vị vật lý chuẩn ASCII (VD: V, A, ohm, J, Hz, %). Để rỗng '' nếu không có đơn vị."
    )


ROW1_FEWSHOT_DEMOS = [
    dspy.Example(
        question="Một học sinh đo thể tích nước được 500 ± 5 ml. Tính sai số tỉ đối phần trăm.",
        reasoning_steps=(
            "Bước 1: Giá trị trung bình của thể tích là V = 500 ml và sai số tuyệt đối là dV = 5 ml.\n"
            "Bước 2: Áp dụng công thức tính sai số tỉ đối phần trăm: delta = (dV / V) * 100%.\n"
            "Bước 3: Thay số vào công thức: delta = (5 / 500) * 100 = 1.\n"
            "Bước 4: Vậy sai số tỉ đối phần trăm của phép đo là 1 %."
        ),
        explanation="Sai số tỉ đối của phép đo được xác định bằng tỉ số phần trăm giữa sai số tuyệt đối và giá trị trung bình theo công thức $\\delta = \\frac{\\Delta V}{\\bar{V}} \\times 100\\%$.",
        answer="1",
        unit="%",
    ).with_inputs("question"),
    dspy.Example(
        question="Mạch RLC nối tiếp đang xảy ra cộng hưởng có điện trở thuần R = 20 Ω. Tổng trở của mạch là bao nhiêu?",
        reasoning_steps=(
            "Bước 1: Đề bài cho mạch RLC nối tiếp ở trạng thái cộng hưởng với điện trở thuần R = 20 ohm.\n"
            "Bước 2: Khi xảy ra cộng hưởng điện (Z_L = Z_C), công thức tổng trở của mạch là Z = sqrt(R^2 + (Z_L - Z_C)^2) = R.\n"
            "Bước 3: Thay số: Z = R = 20.\n"
            "Bước 4: Kết luận tổng trở của mạch khi cộng hưởng là 20 ohm."
        ),
        explanation="Khi mạch RLC nối tiếp xảy ra hiện tượng cộng hưởng điện ($Z_L = Z_C$), tổng trở của mạch đạt giá trị cực tiểu và bằng đúng điện trở thuần: $Z = \\sqrt{R^2 + (Z_L - Z_C)^2} = R$.",
        answer="20",
        unit="ohm",
    ).with_inputs("question"),
    dspy.Example(
        question="Một đoạn mạch nối tiếp có hiệu điện thế hiệu dụng U = 100 V và cường độ dòng điện I = 2 A. Tính tổng trở Z của mạch.",
        reasoning_steps=(
            "Bước 1: Đề bài cho hiệu điện thế U = 100 V và cường độ dòng điện I = 2 A (đã ở đơn vị chuẩn SI).\n"
            "Bước 2: Áp dụng định luật Ohm cho đoạn mạch: Z = U / I.\n"
            "Bước 3: Thay số vào công thức: Z = 100 / 2 = 50.\n"
            "Bước 4: Vậy tổng trở của đoạn mạch là 50 ohm."
        ),
        explanation="Theo định luật Ohm cho toàn mạch, tổng trở của đoạn mạch bằng tỉ số giữa hiệu điện thế hiệu dụng và cường độ dòng điện hiệu dụng: $Z = \\frac{U}{I}$.",
        answer="50",
        unit="ohm",
    ).with_inputs("question"),
    dspy.Example(
        question="Trong một mạch dao động LC lý tưởng, năng lượng dao động toàn phần là 0.08 J. Khi năng lượng điện trường là 0.02 J thì năng lượng từ trường là bao nhiêu?",
        reasoning_steps=(
            "Bước 1: Năng lượng điện từ toàn phần W = 0.08 J, năng lượng điện trường W_d = 0.02 J.\n"
            "Bước 2: Theo định luật bảo toàn năng lượng trong mạch LC lý tưởng: W = W_d + W_t => W_t = W - W_d.\n"
            "Bước 3: Thay số: W_t = 0.08 - 0.02 = 0.06.\n"
            "Bước 4: Vậy năng lượng từ trường trong cuộn cảm là 0.06 J."
        ),
        explanation="Theo định luật bảo toàn năng lượng điện từ trong mạch dao động LC lý tưởng, năng lượng toàn phần bằng tổng năng lượng điện trường và năng lượng từ trường: $W = W_{\\text{đ}} + W_{\\text{t}} \\Rightarrow W_{\\text{t}} = W - W_{\\text{đ}}$.",
        answer="0.06",
        unit="J",
    ).with_inputs("question"),
]


class Row1DirectCoTSolver(dspy.Module):
    """Module Row 1: Qwen2.5-7B tự gánh toàn bộ đọc hiểu, nhớ công thức, tính nhẩm và giải thích trong 1 lượt."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(Row1DirectCoTSignature)
        self.predictor.demos = list(ROW1_FEWSHOT_DEMOS)

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
    return "" if u_str in ("-", "''", '""', "None") else u_str


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
        if getattr(dspy.settings, "lm", None) is None:
            from optimize_prompt import setup_lm
            setup_lm()
        solver = Row1DirectCoTSolver()
        logger.info("[Row 1] Khởi tạo Row1DirectCoTSolver kèm 4 few-shot demos thành công!")
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
