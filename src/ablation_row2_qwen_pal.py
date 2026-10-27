"""
ablation_row2_qwen_pal.py - Cấu hình Ablation Row 2: Qwen + PAL (Mục 2.3, không có RAG & chưa tách Explainer)

Đặc điểm cấu hình Row 2:
- CÓ dùng Symbol-Masking Translation (dịch đề bài sang tiếng Anh Q_en bảo toàn ký hiệu).
- CÓ dùng PAL & Sandbox (Qwen2.5-7B sinh mã Python -> thực thi trong Sandbox lấy final_answer và unit chuẩn).
- KHÔNG dùng Hybrid RAG (không có ngữ cảnh lý thuyết Feynman K, mô hình tự nhớ công thức).
- KHÔNG tách rời Explainer (Qwen phải vừa sinh python_code vừa tự viết trường explanation bằng TIẾNG VIỆT trong cùng 1 lượt sinh).
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Ablation_Row2_QwenPAL")


# ============================================================
# 1. PROMPT & 4 FEW-SHOT DEMOS CHO ROW 2 (KẾ THỪA TỪ OPTIMIZED PROMPTS)
# ============================================================

class Row2JointPALSignature(dspy.Signature):
    """
    You are an expert in Physics, physics education, and Python programming following the Program-Aided Language (PAL) approach.
    Read the physics question carefully (no external RAG context is provided; recall the relevant physical laws yourself).
    You must perform two tasks simultaneously in a single pass:
    (A) Write a concise theoretical explanation in VIETNAMESE into 'explanation'.
    (B) Generate clean, executable Python code into 'python_code' to solve the problem numerically.

    Instructions for 'explanation' (MANDATORY VIETNAMESE - Criterion P2):
    1. Although the input question is in English, 'explanation' MUST BE WRITTEN IN VIETNAMESE (1 to 3 concise sentences).
    2. State the core physical phenomenon and the governing physical law/principle applied.
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
    question: str = dspy.InputField(
        desc="Physics question to solve (translated into English with preserved symbols)."
    )
    explanation: str = dspy.OutputField(
        desc="Đoạn văn ngắn (1-3 câu) BẮT BUỘC BẰNG TIẾNG VIỆT giải thích hiện tượng, nêu định luật và công thức LaTeX ($...$)."
    )
    python_code: str = dspy.OutputField(
        desc="Clean executable Python code defining final_answer and unit. No markdown fences."
    )


DEFAULT_DEMO_EXPLANATIONS = [
    "Sai số tỉ đối của phép đo được xác định bằng tỉ số phần trăm giữa sai số tuyệt đối và giá trị thực tế theo công thức $\\delta = \\frac{\\Delta V}{V} \\times 100\\%$.",
    "Khi mạch RLC nối tiếp xảy ra hiện tượng cộng hưởng điện ($Z_L = Z_C$), tổng trở của mạch đạt giá trị cực tiểu và bằng đúng điện trở thuần: $Z = \\sqrt{R^2 + (Z_L - Z_C)^2} = R$.",
    "Theo định luật Ohm cho toàn mạch, tổng trở của đoạn mạch bằng tỉ số giữa hiệu điện thế hiệu dụng và cường độ dòng điện hiệu dụng: $Z = \\frac{U}{I}$.",
    "Theo định luật bảo toàn năng lượng điện từ trong mạch dao động LC lý tưởng, năng lượng toàn phần bằng tổng năng lượng điện trường và năng lượng từ trường: $W = W_{\\text{đ}} + W_{\\text{t}} \\Rightarrow W_{\\text{t}} = W - W_{\\text{đ}}$.",
]


def _load_row2_demos() -> List[dspy.Example]:
    """Tải 2 demos chuẩn từ optimized_prompt.json (lược bỏ context, gắn thêm explanation tiếng Việt)."""
    demos = []
    if os.path.exists(config.OPTIMIZED_PROMPT_PATH):
        try:
            with open(config.OPTIMIZED_PROMPT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_demos = data.get("predictor", {}).get("demos", [])
            for idx, d in enumerate(raw_demos[:2]):
                exp_vi = DEFAULT_DEMO_EXPLANATIONS[idx] if idx < len(DEFAULT_DEMO_EXPLANATIONS) else DEFAULT_DEMO_EXPLANATIONS[0]
                demos.append(
                    dspy.Example(
                        question=d.get("question", ""),
                        explanation=exp_vi,
                        python_code=d.get("python_code", ""),
                    ).with_inputs("question")
                )
        except Exception as e:
            logger.warning(f"[Row 2] Không thể đọc demos từ {config.OPTIMIZED_PROMPT_PATH}: {e}")
    return demos


class Row2JointPALSolver(dspy.Module):
    """Module Row 2: Sinh đồng thời explanation + python_code trong 1 lượt mà KHÔNG có RAG context."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(Row2JointPALSignature)
        demos = _load_row2_demos()
        if demos:
            self.predictor.demos = demos

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
    return "" if u_str in ("-", "''", '""', "None") else u_str


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
        if getattr(dspy.settings, "lm", None) is None:
            from optimize_prompt import setup_lm
            setup_lm()
        solver = Row2JointPALSolver()
        logger.info(f"[Row 2] Khởi tạo Row2JointPALSolver kèm {len(solver.predictor.demos)} demos thành công!")
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
