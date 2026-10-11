"""
metric.py - Hàm chấm điểm PAL cho DSPy ViPRA 2026
Chạy python_code qua exec(), lấy final_answer, so sánh với ground truth.
"""
import re
import math
from config import NUMERICAL_TOLERANCE


# ============================================================
# BƯỚC 1: Làm sạch code trước khi exec()
# ============================================================
def clean_python_code(raw_code: str) -> str:
    """
    Làm sạch chuỗi code thô từ LLM trước khi đưa vào exec().
    Xử lý các lỗi phổ biến mà LLM 7B hay mắc:
    - Bọc trong markdown fence: ```python ... ```
    - Dùng ký tự Unicode thay vì ASCII Python
    - Dòng trống thừa ở đầu/cuối
    """
    code = raw_code.strip()

    # Bóc markdown fence (```python ... ``` hoặc ``` ... ```)
    code = re.sub(r"^```(?:python)?\s*\n?", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\n?```\s*$", "", code)
    code = code.strip()

    # Thay ký tự Unicode toán học → ASCII Python
    replacements = {
        "×": "*",
        "·": "*",
        "÷": "/",
        "−": "-",       # dấu trừ Unicode
        "–": "-",       # en dash
        "μ": "1e-6*",   # micro (dùng trong biểu thức trực tiếp)
        "Ω": "ohm",     # ohm → sẽ là string, không tính được nhưng tránh crash
        "π": "math.pi",
        "²": "**2",
        "³": "**3",
        "^": "**",      # LLM hay dùng ^ thay **
    }
    for old, new in replacements.items():
        code = code.replace(old, new)

    return code


# ============================================================
# BƯỚC 2: Chạy code và trích xuất final_answer
# ============================================================
def execute_code(code: str) -> tuple[any, str]:
    """
    Chạy code trong môi trường cô lập, trả về (final_answer, unit).
    Trả về (None, "") nếu có lỗi.
    """
    namespace = {
        "__builtins__": {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "range": range, "list": list,
            "float": float, "int": int, "str": str, "print": lambda *a: None,
        },
        "math": math,
        # Hằng số vật lý hay dùng
        "pi": math.pi,
        "e": math.e,
        "k": 9e9,           # Hằng số Coulomb
        "eps0": 8.854e-12,  # Hằng số điện môi chân không
        "mu0": 4 * math.pi * 1e-7,  # Hằng số từ thẩm chân không
    }

    try:
        exec(compile(code, "<llm_code>", "exec"), namespace)
    except SyntaxError:
        return None, ""
    except Exception:
        return None, ""

    final_answer = namespace.get("final_answer", None)
    unit = str(namespace.get("unit", "")).strip()

    # Chuẩn hóa unit: "-" → ""
    if unit == "-":
        unit = ""

    return final_answer, unit


# ============================================================
# BƯỚC 3: So sánh đáp án
# ============================================================
def _normalize_unit(unit: str) -> str:
    return unit.strip().lower().replace(" ", "")


def _compare_answers(pred_val: any, true_str: str) -> bool:
    """
    So sánh giá trị từ exec() với đáp án chuẩn (chuỗi từ CSV).
    Hỗ trợ: số đơn, list (bài nhiều phần), văn bản.
    """
    # Thử parse ground truth thành số
    try:
        true_num = float(true_str.strip())
    except (ValueError, AttributeError):
        true_num = None

    if true_num is not None and pred_val is not None:
        try:
            pred_num = float(pred_val)
            return abs(pred_num - true_num) <= NUMERICAL_TOLERANCE
        except (ValueError, TypeError):
            pass

    # So khớp văn bản (Có / Không / Cộng hưởng...)
    return str(pred_val).strip().lower() == str(true_str).strip().lower()


def _compare_multi_part(pred_val: any, true_answer: str,
                        pred_unit: str, true_unit: str) -> bool:
    """
    Xử lý bài nhiều phần:
    - pred_val có thể là list: [0.50, 1.25]
    - true_answer là chuỗi: "0.50; 1.25"
    """
    true_parts = [p.strip() for p in true_answer.split(";")]
    true_units = [p.strip() for p in true_unit.split(";")]
    pred_units = [p.strip() for p in pred_unit.split(";")]

    # Chuẩn hóa pred_val thành list
    if isinstance(pred_val, (list, tuple)):
        pred_parts = [str(v) for v in pred_val]
    else:
        pred_parts = [str(pred_val)]

    if len(pred_parts) != len(true_parts):
        return False

    for i, (p, t) in enumerate(zip(pred_parts, true_parts)):
        p_unit = pred_units[i] if i < len(pred_units) else ""
        t_unit = true_units[i] if i < len(true_units) else ""

        if _normalize_unit(p_unit) != _normalize_unit(t_unit):
            return False
        try:
            if not _compare_answers(float(p), t):
                return False
        except (ValueError, TypeError):
            if not _compare_answers(p, t):
                return False

    return True


# ============================================================
# BƯỚC 4: Hàm metric chính cho DSPy Optimizer
# ============================================================
def btc_correctness_metric(example: object, pred: object, trace=None) -> float:
    """
    Metric PAL cho DSPy:
    1. Làm sạch python_code từ LLM
    2. exec() trong môi trường cô lập → lấy final_answer, unit
    3. So sánh với example.answer + example.unit theo đúng luật BTC
       (sai số tuyệt đối 0.05, khớp unit ASCII)
    Trả về 1.0 nếu đúng, 0.0 nếu sai hoặc có lỗi.
    """
    try:
        raw_code = str(pred.python_code)
        code = clean_python_code(raw_code)

        final_answer, pred_unit = execute_code(code)

        if final_answer is None:
            return 0.0

        true_answer = str(example.answer).strip()
        true_unit   = str(example.unit).strip()
        if true_unit == "-":
            true_unit = ""

        # Bài nhiều phần
        if ";" in true_answer or isinstance(final_answer, (list, tuple)):
            return 1.0 if _compare_multi_part(
                final_answer, true_answer, pred_unit, true_unit
            ) else 0.0

        # Bài đơn
        if _normalize_unit(pred_unit) != _normalize_unit(true_unit):
            return 0.0

        return 1.0 if _compare_answers(final_answer, true_answer) else 0.0

    except Exception:
        return 0.0
