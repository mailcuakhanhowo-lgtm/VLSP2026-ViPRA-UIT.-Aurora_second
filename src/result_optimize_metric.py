"""
metric.py - Hàm chấm điểm PAL cho DSPy ViPRA 2026
Chạy python_code qua exec(), lấy final_answer, so sánh với ground truth.
"""
import re
import math
import concurrent.futures
try:
    import numpy as np
except ImportError:
    np = None
try:
    import sympy as sp
except ImportError:
    sp = None
from config import NUMERICAL_TOLERANCE, SANDBOX_TIMEOUT_SECONDS


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

    # Loại bỏ dòng import math / cmath thừa vì đã được nạp sẵn vào namespace
    code = re.sub(r"^\s*(?:import\s+(?:math|cmath)|from\s+(?:math|cmath)\s+import\s+.*)\s*$", "", code, flags=re.MULTILINE)

    return code


# ============================================================
# BƯỚC 2: Chạy code và trích xuất final_answer với Timeout 15s
# ============================================================
def execute_code(code: str, timeout: float = None) -> tuple[any, str]:
    """
    Chạy code trong môi trường cô lập có Timeout cứng (mặc định 15s).
    Trả về (final_answer, unit). Trả về (None, "") nếu lỗi hoặc hết thời gian.
    """
    if timeout is None:
        timeout = float(SANDBOX_TIMEOUT_SECONDS)

    def _safe_import(name, *args, **kwargs):
        if name in ("math", "cmath", "decimal", "fractions", "numpy", "scipy", "sympy"):
            import importlib
            return importlib.import_module(name)
        raise ImportError(f"Import {name} is restricted in sandbox")

    namespace = {
        "__builtins__": {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "range": range, "list": list,
            "float": float, "int": int, "str": str, "print": lambda *a: None,
            "__import__": _safe_import,
        },
        "math": math,
        "np": np if np is not None else math,
        "numpy": np if np is not None else math,
        "sp": sp,
        "sympy": sp,
        "t": 0,  # Thời điểm ban đầu t=0 cho biểu thức dao động/xoay chiều nếu LLM chưa gán t
        # Lượng giác tính theo ĐỘ (giúp LLM không cần đổi radians thủ công)
        "sin_deg": lambda x: math.sin(math.radians(x)),
        "cos_deg": lambda x: math.cos(math.radians(x)),
        "tan_deg": lambda x: math.tan(math.radians(x)),
        # Hằng số vật lý tiêu chuẩn BTC
        "pi": math.pi,
        "e": math.e,
        "k": 9e9,           # Hằng số Coulomb
        "eps0": 8.854e-12,  # Hằng số điện môi chân không
        "mu0": 4 * math.pi * 1e-7,  # Hằng số từ thẩm chân không
        "g": 9.8,           # Gia tốc trọng trường chuẩn
    }

    try:
        compiled_code = compile(code, "<llm_code>", "exec")
    except SyntaxError:
        return None, ""
    except Exception:
        return None, ""

    import threading

    res = [None, ""]
    err = [None]

    def _worker():
        max_retries = 3
        current_err = None
        for _ in range(max_retries):
            try:
                exec(compiled_code, namespace)
                ans = namespace.get("final_answer", None)
                u = str(namespace.get("unit", "")).strip()
                if u == "-":
                    u = ""
                # Chuyển đổi an toàn nếu ans là đối tượng SymPy (Float, Rational, Expr...)
                if ans is not None and hasattr(ans, "evalf"):
                    try:
                        ans = float(ans.evalf())
                    except Exception:
                        ans = str(ans)
                res[0] = ans
                res[1] = u
                return
            except NameError as ne:
                current_err = ne
                missing_var = getattr(ne, "name", None)
                if not missing_var and "name '" in str(ne):
                    m = re.search(r"name '(\w+)' is not defined", str(ne))
                    if m:
                        missing_var = m.group(1)
                if missing_var and missing_var not in namespace:
                    if sp is not None:
                        namespace[missing_var] = sp.Symbol(missing_var)
                    else:
                        namespace[missing_var] = 0
                else:
                    break
            except Exception as e:
                current_err = e
                break
        err[0] = current_err

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"[Sandbox Warning] Code chay vuot qua {timeout}s! Da ngat tien trinh.")
        return None, ""

    if err[0] is not None:
        return None, ""

    return res[0], res[1]


# ============================================================
# BƯỚC 3: So sánh đáp án
# ============================================================
def _normalize_unit(unit: str) -> str:
    """Chuẩn hóa đơn vị Canonical: xử lý lũy thừa J/m**3 -> J/m^3, N/C == V/m, ohm."""
    u = str(unit).strip().lower().replace(" ", "")
    # Chuẩn hóa số mũ
    u = u.replace("**", "^")
    # Chuẩn hóa điện trở
    u = u.replace(r"\omega", "ohm").replace("ω", "ohm").replace("omega", "ohm")
    # Chuẩn hóa cường độ điện trường V/m tương đương N/C
    if u in ("n/c", "v/m"):
        u = "v/m"
    if u in ("-", "none", "null", ""):
        u = ""
    return u


def _compare_answers(pred_val: any, true_str: str) -> bool:
    """
    So sánh giá trị từ exec() với đáp án chuẩn (chuỗi từ CSV).
    Hỗ trợ: sai số tuyệt đối <= 0.05 HOẶC sai số tương đối <= 5% (0.05).
    """
    # Thử parse ground truth thành số
    try:
        true_num = float(true_str.strip())
    except (ValueError, AttributeError):
        true_num = None

    if true_num is not None and pred_val is not None:
        try:
            pred_num = float(pred_val)
            abs_diff = abs(pred_num - true_num)
            # 1. Sai số tuyệt đối <= NUMERICAL_TOLERANCE (mặc định 0.05)
            if abs_diff <= NUMERICAL_TOLERANCE:
                return True
            # 2. Sai số tương đối <= 5% (NUMERICAL_TOLERANCE)
            if abs(true_num) > 1e-12 and (abs_diff / abs(true_num)) <= NUMERICAL_TOLERANCE:
                return True
            return False
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
