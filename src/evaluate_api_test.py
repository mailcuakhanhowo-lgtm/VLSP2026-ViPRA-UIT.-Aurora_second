"""
evaluate_api_test.py - Bộ công cụ kiểm thử & chấm điểm Live HTTP API chuẩn ViPRA 2026.
Mục tiêu:
1. Nạp danh sách câu hỏi kiểm thử từ datasets/test_dataset.csv.
2. Giả lập Server chấm điểm của BTC: Gửi POST tuần tự từng câu hỏi vào Endpoint (/predict) với Timeout 60s.
3. Thu nhận và kiểm tra định dạng phản hồi JSON chuẩn BTC 5 trường (query_id, answer, unit, explanation, reasoning).
4. Tự động chấm điểm 100% P1 (Correctness):
   - So khớp giá trị số với dung sai 0.05 (kết hợp cả sai số tuyệt đối <= 0.05 và sai số tương đối <= 5%).
   - Xử lý biểu thức toán học, ký hiệu khoa học (1.25*10^7, sqrt, phân số).
   - Xử lý câu hỏi nhiều phần (phân tách bởi '; ').
   - Chuẩn hóa đơn vị đo lường Canonical ASCII (V/m == N/C, ohm == \Omega/ω).
   - Kiểm tra tính hợp lệ của trường explanation (không được để trống).
5. Xuất báo cáo tổng kết chi tiết và lưu file kết quả JSON.
"""

import os
import sys
import re
import math
import time
import json
import argparse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Đảm bảo console Windows in Unicode tiếng Việt không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thiết lập đường dẫn gốc dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import config


# ============================================================
# 1. HÀM CHUẨN HÓA VÀ SO KHỚP ĐƠN VỊ & GIÁ TRỊ TOÁN HỌC
# ============================================================

def normalize_unit(unit_str: Any) -> str:
    """
    Chuẩn hóa đơn vị đo lường vật lý:
    - Loại bỏ dấu $, khoảng trắng, chuyển chữ thường.
    - Loại bỏ các wrapper LaTeX: \\text{...}, \\mathrm{...}, \\mathbf{...}.
    - Chuẩn hóa ký tự micro: \\mu, μ (U+03BC), µ (U+00B5) -> u (uF, uC, uJ, uH).
    - Chuẩn hóa điện trở: \\omega, \\Omega, omega, ohm, Ω, ôm -> ohm.
    - Chuẩn hóa cường độ điện trường: N/C == V/m -> v/m.
    - Chuẩn hóa phần trăm: \\% -> %.
    - Chuẩn hóa số mũ: ** -> ^, loại bỏ ngoặc {} trong số mũ (m^{3} -> m^3).
    - Chuẩn hóa rỗng / không thứ nguyên: '-' / 'none' / 'null' / '' / 'không' -> ''.
    """
    if unit_str is None:
        return ""
    u = str(unit_str).strip().lower()
    # Loại bỏ math delimiters LaTeX
    u = u.replace("$", "").replace("\\$", "")
    # Loại bỏ \\text{} và \\mathrm{}
    u = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]+)\}", r"\1", u)
    u = u.replace(" ", "")
    u = u.replace("**", "^")
    # Bỏ dấu ngoặc trong số mũ (VD: m^{3} -> m^3)
    u = re.sub(r"\^{([^}]+)\}", r"^\1", u)
    # Chuẩn hóa ký hiệu LaTeX \\mu và ký tự Unicode micro
    u = u.replace(r"\mu", "u").replace("\u03bc", "u").replace("\u00b5", "u")
    # Chuẩn hóa ký hiệu LaTeX \\Omega / omega
    u = u.replace(r"\omega", "ohm").replace("ω", "ohm").replace("omega", "ohm").replace("Ω", "ohm").replace("ôm", "ohm")
    # Chuẩn hóa ký hiệu LaTeX \\%
    u = u.replace(r"\%", "%")
    # Chuẩn hóa cường độ điện trường
    if u in ("n/c", "v/m"):
        u = "v/m"
    # Chuẩn hóa không có đơn vị
    if u in ("-", "none", "null", "", "không", "khong"):
        u = ""
    return u


def parse_unit_and_scale(unit_str: str) -> Tuple[str, float]:
    """
    Tách đơn vị thành (đơn vị cơ bản, hệ số quy đổi về SI).
    VD:
    'nc' -> ('c', 1e-9)
    'mj' -> ('j', 1e-3)
    'uf' -> ('f', 1e-6)
    'pf' -> ('f', 1e-12)
    'mh' -> ('h', 1e-3)
    'kohm' -> ('ohm', 1e3)
    'c' -> ('c', 1.0)
    'v/m' -> ('v/m', 1.0)
    """
    u = normalize_unit(unit_str).strip().lower()
    if not u:
        return "", 1.0

    # Các trường hợp đặc biệt không tách prefix
    if u in ("v/m", "%", "ohm"):
        return u, 1.0

    # Bảng tiền tố SI chuẩn
    prefixes = [
        ("p", 1e-12),   # pico (pF, pC...)
        ("n", 1e-9),    # nano (nC, nF, nH...)
        ("u", 1e-6),    # micro (uF, uC, uH, uJ, uA, uV...)
        ("m", 1e-3),    # milli (mJ, mH, mV, mA, mC, mm, ms...)
        ("c", 1e-2),    # centi (cm...)
        ("k", 1e3),     # kilo (kV, kA, kJ, kW, kHz, kohm, km...)
        ("M", 1e6),     # mega (MW, MHz, Mohm...)
    ]

    known_bases = ("ohm", "hz", "wb", "v", "a", "c", "j", "w", "f", "h", "t", "n", "m", "s")

    for p_char, scale in prefixes:
        if u.startswith(p_char.lower()):
            candidate_base = u[len(p_char):]
            if candidate_base in known_bases:
                return candidate_base, scale

    return u, 1.0


def parse_numeric_value(val_str: Any) -> Optional[float]:
    """
    Chuyển đổi chuỗi giá trị toán học thành số float:
    - Hỗ trợ số thập phân dấu chấm (1.25) và dấu phẩy tiếng Việt (1,25).
    - Hỗ trợ ký hiệu khoa học đa dạng: '1.25*10^7', '1.25 \\times 10^{7}', '1.25x10^7', '10^-6'.
    - Hỗ trợ căn bậc hai và phân số: 'sqrt(61)', '1/2', '\\frac{1}{2}'.
    - KHÔNG gán 'e' thành hằng số Euler để bảo vệ biểu thức đại số chứa biến E (như 8*E).
    """
    if val_str is None:
        return None
    if isinstance(val_str, (int, float)):
        return float(val_str)

    s = str(val_str).strip()
    if not s:
        return None

    # Loại bỏ dấu math delimiters LaTeX và dấu gạch ngang rác / chấm câu ở cuối
    s = s.replace("$", "").replace("\\$", "").strip()
    s = s.rstrip(" .–-").strip()

    # Bỏ các tiền tố xấp xỉ LaTeX hoặc text
    s = re.sub(r"^\\(?:approx|sim)\s*", "", s)
    s = re.sub(r"^[≈~]\s*", "", s)

    # Xử lý phân số LaTeX: \\frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)

    # 1. Thử parse trực tiếp số float thông thường
    try:
        return float(s)
    except ValueError:
        pass

    # 2. Thử chuyển dấu phẩy thập phân kiểu Việt Nam (VD: "1,25" -> "1.25")
    # Chỉ thay thế khi dấu phẩy đứng giữa các chữ số và không có dấu chấm
    if re.match(r"^[+-]?\d+,\d+$", s):
        try:
            return float(s.replace(",", "."))
        except ValueError:
            pass

    # 3. Chuẩn hóa các dạng nhân và lũy thừa 10
    # Thay thế các dạng dấu nhân: \\times, \\cdot, ×, x, X
    s_norm = s.replace(" ", "")
    s_norm = s_norm.replace(r"\times", "*").replace(r"\cdot", "*").replace("×", "*").replace("·", "*")
    s_norm = s_norm.replace("–", "-").replace("−", "-")
    # Loại bỏ ngoặc nhọn trong số mũ: 10^{7} -> 10^7, 10^{-3} -> 10^-3
    s_norm = re.sub(r"10\^{([+-]?\d+)\}", r"10^\1", s_norm)
    # Hỗ trợ dạng 1.25x10^7 hoặc 1.25X10^7
    s_norm = re.sub(r"(\d)[xX](10\^[+-]?\d+)", r"\1*\2", s_norm)

    # Bắt dạng base * 10^exp
    sci_match = re.match(r"^([+-]?\d+(?:\.\d+)?)\*10\^([+-]?\d+)$", s_norm, re.IGNORECASE)
    if sci_match:
        try:
            base = float(sci_match.group(1))
            exp = int(sci_match.group(2))
            return base * (10 ** exp)
        except Exception:
            pass

    # Bắt dạng chỉ có 10^exp (VD: 10^7, 10^-3)
    pure_exp_match = re.match(r"^([+-]?)10\^([+-]?\d+)$", s_norm, re.IGNORECASE)
    if pure_exp_match:
        try:
            sign = -1.0 if pure_exp_match.group(1) == "-" else 1.0
            exp = int(pure_exp_match.group(2))
            return sign * (10 ** exp)
        except Exception:
            pass

    # 4. Tính toán an toàn cho biểu thức toán học (sqrt, phân số, pi)
    # LƯU Ý: KHÔNG đưa 'e' vào safe_dict để tránh nhận nhầm biến E trong vật lý thành Euler 2.718
    safe_dict = {
        "sqrt": math.sqrt,
        "pi": math.pi,
        "abs": abs,
    }
    expr = s_norm.replace("^", "**")
    # Chỉ cho phép các ký tự số học và hàm trong safe_dict (KHÔNG có chữ e đứng lẻ ngoài khoa học)
    if re.match(r"^[0-9\.\+\-\*\/\(\)\s]+$", expr) or any(fn in expr for fn in ["sqrt", "pi", "abs"]):
        clean_expr_check = re.sub(r"\b(sqrt|pi|abs)\b", "", expr)
        if re.match(r"^[0-9\.\+\-\*\/\(\)\s]+$", clean_expr_check):
            try:
                val = eval(expr, {"__builtins__": None}, safe_dict)
                return float(val)
            except Exception:
                pass

    return None


def compare_single_value(pred_val: Any, true_val: Any, tolerance: float = 0.05) -> Tuple[bool, str]:
    """
    So sánh một cặp giá trị (Dự đoán vs Ground Truth):
    - Đúng nếu sai số tuyệt đối <= tolerance (cho số nhỏ xung quanh 0).
    - Hoặc sai số tương đối <= tolerance (5%) (cho số lớn như 10^7 hoặc số rất nhỏ như 10^-6).
    - Xử lý biểu thức đại số như '8 * E' == '8*E' == '8E'.
    - Xử lý chuỗi văn bản tiếng Việt ('Tăng 4 lần' == 'Tăng gấp 4 lần', 'Có' == 'Có.').
    """
    pred_num = parse_numeric_value(pred_val)
    true_num = parse_numeric_value(true_val)

    if pred_num is not None and true_num is not None:
        abs_diff = abs(pred_num - true_num)
        # 1. Sai số tuyệt đối <= 0.05
        if abs_diff <= tolerance:
            return True, f"Khớp số tuyệt đối (diff={abs_diff:.6g} <= {tolerance})"

        # 2. Sai số tương đối <= 5% (dung sai 0.05 theo chuẩn kỹ thuật)
        if abs(true_num) > 1e-12:
            rel_error = abs_diff / abs(true_num)
            if rel_error <= tolerance:
                return True, f"Khớp sai số tương đối (rel_error={rel_error*100:.2f}% <= {tolerance*100}%)"

        return False, f"Sai số vượt ngưỡng (pred={pred_num:.6g}, true={true_num:.6g}, abs_diff={abs_diff:.6g})"

    # Chuẩn hóa chuỗi văn bản
    p_str = str(pred_val).strip().lower().rstrip(" .–-").strip()
    t_str = str(true_val).strip().lower().rstrip(" .–-").strip()

    # So sánh chính xác chuỗi
    if p_str == t_str:
        return True, "Khớp chuỗi văn bản trực tiếp"

    # So sánh biểu thức đại số loại trừ khoảng trắng và dấu nhân (VD: "8 * E" == "8*E" == "8E")
    p_norm_expr = p_str.replace(" ", "").replace("*", "").replace(r"\times", "").replace(r"\cdot", "")
    t_norm_expr = t_str.replace(" ", "").replace("*", "").replace(r"\times", "").replace(r"\cdot", "")
    if p_norm_expr == t_norm_expr and len(p_norm_expr) > 0:
        return True, "Khớp biểu thức đại số ký hiệu"

    # So sánh tương đương ngữ nghĩa tiếng Việt phổ biến (VD: "tăng gấp 4 lần" == "tăng 4 lần")
    p_semantic = p_str.replace("gấp", "").replace(" ", "").replace(".", "")
    t_semantic = t_str.replace("gấp", "").replace(" ", "").replace(".", "")
    if p_semantic == t_semantic and len(p_semantic) > 0:
        return True, f"Khớp ngữ nghĩa câu trả lời tiếng Việt ('{p_str}' ~ '{t_str}')"

    # So sánh khẳng định "có" / "đúng"
    if p_str in ("có", "dung", "đúng", "yes") and t_str in ("có", "dung", "đúng", "yes"):
        return True, "Khớp khẳng định logic"

    return False, f"Lệch chuỗi văn bản: pred='{p_str}' != true='{t_str}'"


def grade_prediction(
    pred_data: Dict[str, Any],
    true_answer: str,
    true_unit: str,
    expected_query_id: Optional[str] = None,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Chấm điểm 1 dự đoán theo toàn bộ tiêu chí P1 của BTC:
    - Kiểm tra query_id khớp với đầu vào (Mục 4.2).
    - So khớp answer (hỗ trợ bài nhiều phần '; ').
    - So khớp unit (chuẩn hóa canonical ASCII).
    - Kiểm tra tính đầy đủ của trường explanation (bắt buộc theo Mục 4.2 & 4.5).
    Trả về dict chi tiết kết quả chấm.
    """
    pred_raw_answer = pred_data.get("answer", "")
    pred_raw_unit = pred_data.get("unit", "")
    explanation = str(pred_data.get("explanation", "")).strip()
    pred_query_id = str(pred_data.get("query_id", "")).strip()

    # 1. Kiểm tra query_id nếu có
    query_id_match = True
    if expected_query_id:
        query_id_match = (pred_query_id == str(expected_query_id).strip())

    # 2. Kiểm tra trường explanation bắt buộc theo Mục 4.2 & 4.5
    has_explanation = bool(explanation and explanation != "None" and len(explanation) >= 5)

    # 3. Xử lý bài toán nhiều phần (Multi-part)
    true_ans_parts = [p.strip() for p in str(true_answer).split(";")]
    true_unit_parts = [p.strip() for p in str(true_unit).split(";")]
    pred_ans_parts = [p.strip() for p in str(pred_raw_answer).split(";")]
    pred_unit_parts = [p.strip() for p in str(pred_raw_unit).split(";")]

    # Đảm bảo số phần của unit khớp với số phần của answer
    while len(true_unit_parts) < len(true_ans_parts):
        true_unit_parts.append("")
    while len(pred_unit_parts) < len(pred_ans_parts):
        pred_unit_parts.append("")

    if len(pred_ans_parts) != len(true_ans_parts):
        return {
            "p1_correct": False,
            "query_id_match": query_id_match,
            "answer_correct": False,
            "unit_correct": False,
            "has_explanation": has_explanation,
            "reason": f"Lệch số phần: Pred có {len(pred_ans_parts)} phần, True có {len(true_ans_parts)} phần.",
            "parts_detail": []
        }

    all_answers_match = True
    all_units_match = True
    parts_detail = []

    for idx, (p_a, t_a, p_u, t_u) in enumerate(zip(pred_ans_parts, true_ans_parts, pred_unit_parts, true_unit_parts)):
        p_base_unit, p_scale = parse_unit_and_scale(p_u)
        t_base_unit, t_scale = parse_unit_and_scale(t_u)

        pred_num = parse_numeric_value(p_a)
        true_num = parse_numeric_value(t_a)

        ans_ok = False
        unit_ok = False
        ans_reason = ""

        # Nếu cả 2 đều là số và cùng loại đơn vị vật lý (hoặc cùng rỗng)
        if pred_num is not None and true_num is not None and (p_base_unit == t_base_unit or not t_base_unit or not p_base_unit):
            # Quy đổi cả 2 về chuẩn SI để so sánh sai số 5%
            p_val_si = pred_num * p_scale
            t_val_si = true_num * t_scale

            abs_diff = abs(p_val_si - t_val_si)
            # Cho phép sai số tương đối <= 5% (tolerance = 0.05) HOẶC sai số tuyệt đối <= 0.05 * t_scale
            if abs(t_val_si) > 1e-15:
                rel_error = abs_diff / abs(t_val_si)
                if rel_error <= tolerance:
                    ans_ok = True
                    unit_ok = True
                    ans_reason = f"Khớp sai số tương đối 5% (rel_err={rel_error*100:.2f}% <= {tolerance*100}%)"
            if not ans_ok and abs_diff <= tolerance * t_scale:
                ans_ok = True
                unit_ok = True
                ans_reason = f"Khớp sai số tuyệt đối (diff={abs_diff:.6g} <= {tolerance * t_scale:.6g})"

            if not ans_ok:
                ans_reason = f"Sai số vượt 5% (pred={p_val_si:.6g}, true={t_val_si:.6g}, abs_diff={abs_diff:.6g})"
                norm_p_u = normalize_unit(p_u)
                norm_t_u = normalize_unit(t_u)
                unit_ok = (norm_p_u == norm_t_u or p_base_unit == t_base_unit)
        else:
            # So sánh thông thường (cho chuỗi, biểu thức hoặc đơn vị không đổi)
            ans_ok, ans_reason = compare_single_value(p_a, t_a, tolerance)
            norm_p_u = normalize_unit(p_u)
            norm_t_u = normalize_unit(t_u)
            unit_ok = (norm_p_u == norm_t_u or (p_base_unit == t_base_unit and bool(p_base_unit)))

        if not ans_ok:
            all_answers_match = False
        if not unit_ok:
            all_units_match = False

        parts_detail.append({
            "part": idx + 1,
            "pred_ans": p_a,
            "true_ans": t_a,
            "ans_match": ans_ok,
            "ans_reason": ans_reason,
            "pred_unit": p_u,
            "true_unit": t_u,
            "unit_match": unit_ok,
        })

    # Tiêu chí P1 của BTC:
    # 1. Answer đúng 100% các phần
    # 2. Unit đúng 100% các phần
    # 3. Explanation phải hiện diện và không được rỗng
    # 4. query_id phải khớp với đầu vào
    p1_correct = all_answers_match and all_units_match and has_explanation and query_id_match

    reason = "Chính xác hoàn toàn"
    if not p1_correct:
        reasons = []
        if not all_answers_match:
            reasons.append("Sai lệch giá trị answer")
        if not all_units_match:
            reasons.append("Sai lệch đơn vị unit")
        if not has_explanation:
            reasons.append("Thiếu trường explanation bắt buộc")
        if not query_id_match:
            reasons.append(f"Mismatched query_id ('{pred_query_id}' != '{expected_query_id}')")
        reason = "; ".join(reasons)

    return {
        "p1_correct": p1_correct,
        "query_id_match": query_id_match,
        "answer_correct": all_answers_match,
        "unit_correct": all_units_match,
        "has_explanation": has_explanation,
        "reason": reason,
        "parts_detail": parts_detail
    }


# ============================================================
# 2. HTTP CLIENT GỬI TRUY VẤN MÔ PHỎNG MÁY CHẤM BTC
# ============================================================

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False


def send_query_to_api(
    api_url: str,
    query_id: str,
    query_text: str,
    timeout: float = 60.0,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """
    Gửi 1 truy vấn đơn đến Endpoint (/predict) theo đúng chuẩn máy chấm BTC:
    - Request: POST JSON {"query_id": "...", "query": "..."}
    - Timeout: 60s
    - Output mong đợi: JSON List chứa 1 object [{...}]
    - Sử dụng curl_cffi giả lập TLS Chrome để tránh bị Cloudflare WAF chặn
    Trả về: (dict_ket_qua, thoi_gian_latency, thong_diep_loi)
    """
    payload = {
        "query_id": query_id,
        "query": query_text
    }

    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)

    start_time = time.time()

    # Cách 1: Sử dụng curl_cffi giả lập Chrome 120 (vượt 100% Cloudflare WAF/Tunnel)
    if HAS_CURL_CFFI:
        try:
            resp = cffi_requests.post(
                api_url,
                json=payload,
                headers=req_headers,
                impersonate="chrome120",
                timeout=timeout
            )
            latency = time.time() - start_time
            if resp.status_code == 200:
                resp_json = resp.json()
                if isinstance(resp_json, list) and len(resp_json) > 0:
                    return resp_json[0], latency, ""
                elif isinstance(resp_json, dict):
                    return resp_json, latency, ""
                else:
                    return None, latency, f"Định dạng phản hồi không hợp lệ: {type(resp_json)}"
            else:
                return None, latency, f"HTTP Error Status {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            err_str = str(e)
            # Nếu là lỗi phân giải tên miền hoặc timeout thì ghi nhận và không thử lại
            if "Could not resolve" in err_str or "timed out" in err_str.lower():
                latency = time.time() - start_time
                return None, latency, f"curl_cffi Error: {err_str}"

    # Cách 2: Dự phòng qua urllib chuẩn
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers=req_headers, method="POST")

    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            latency = time.time() - start_time
            if resp.status == 200:
                resp_text = resp.read().decode("utf-8")
                resp_json = json.loads(resp_text)

                if isinstance(resp_json, list) and len(resp_json) > 0:
                    return resp_json[0], latency, ""
                elif isinstance(resp_json, dict):
                    return resp_json, latency, ""
                else:
                    return None, latency, f"Định dạng phản hồi không hợp lệ: {type(resp_json)}"
            else:
                return None, latency, f"HTTP Error Status {resp.status}"

    except urllib.error.HTTPError as he:
        latency = time.time() - start_time
        try:
            err_body = he.read().decode("utf-8")
        except Exception:
            err_body = str(he)
        return None, latency, f"HTTP {he.code}: {err_body[:200]}"
    except urllib.error.URLError as ue:
        latency = time.time() - start_time
        return None, latency, f"Connection/Timeout Error: {ue.reason}"
    except TimeoutError:
        latency = time.time() - start_time
        return None, latency, "TIMEOUT (> 60s quy định)"
    except Exception as e:
        latency = time.time() - start_time
        return None, latency, f"Lỗi không xác định: {str(e)}"


# ============================================================
# 3. KỊCH BẢN CHẠY KIỂM THỬ VÀ ĐÁNH GIÁ CHÍNH
# ============================================================

def run_evaluation(
    api_url: str,
    csv_path: str,
    timeout: float = 60.0,
    limit: Optional[int] = None,
    delay: float = 0.5,
    output_json_path: str = "output/test_evaluation_results.json",
    submission_json_path: str = "output/submission_test.json",
    headers: Optional[Dict[str, str]] = None,
    mock_mode: bool = False,
) -> Dict[str, Any]:
    """
    Thực thi quy trình kiểm thử hoàn chỉnh:
    - Đọc tập test CSV
    - Gửi tuần tự từng câu hỏi
    - Chấm điểm P1
    - Xuất file báo cáo
    """
    print("\n" + "=" * 70)
    print("[+] BAT DAU QUY TRINH KIEM TRA & CHAM DIEM API - ViPRA 2026")
    print("=" * 70)
    print(f"[*] API Target  : {api_url if not mock_mode else '[MOCK SIMULATION MODE]'}")
    print(f"[*] Dataset     : {csv_path}")
    print(f"[*] Timeout     : {timeout} giay/cau")
    print(f"[*] Dung sai P1 : 0.05 (Sai so tuyet doi hoac sai so tuong doi 5%)")

    # 1. Đọc dữ liệu CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Khong tim thay file du lieu: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    if limit and limit > 0:
        df = df.head(limit)
        print(f"[*] Che do TEST NHANH: Chi chay {limit} cau dau tien.")

    total_questions = len(df)
    print(f"[*] Tong so cau hoi can cham: {total_questions} cau.")
    print("=" * 70 + "\n")

    results_detail = []
    submission_list = []
    
    total_correct = 0
    total_answered = 0
    total_timeouts = 0
    total_http_errors = 0
    total_latency = 0.0

    for idx, row in df.iterrows():
        q_num = idx + 1
        q_id = str(row["id"]).strip()
        q_text = str(row["question"]).strip()
        t_ans = str(row["answer"]).strip()
        t_unit = str(row["unit"]).strip() if pd.notna(row["unit"]) else ""

        print(f"[{q_num:02d}/{total_questions:02d}] Gui Query ID: {q_id} ...", end=" ", flush=True)

        if mock_mode:
            # Chế độ Mock: Giả lập trả về đáp án chuẩn để kiểm thử thuật toán chấm điểm
            pred_obj = {
                "query_id": q_id,
                "answer": t_ans,
                "unit": t_unit,
                "explanation": f"Ap dung dinh luat vat ly tinh toan: final_answer = {t_ans} {t_unit}.",
                "reasoning": {"type": "cot", "steps": ["Nhan dien du kien", "Tinh toan", "Ket luan"]}
            }
            latency = 0.05
            err_msg = ""
        else:
            pred_obj, latency, err_msg = send_query_to_api(api_url, q_id, q_text, timeout, headers)

        total_latency += latency

        if err_msg or not pred_obj:
            is_timeout = "TIMEOUT" in err_msg.upper()
            if is_timeout:
                total_timeouts += 1
            else:
                total_http_errors += 1

            print(f"[FAIL] ({latency:.2f}s) - {err_msg}")
            
            # Ghi nhận trường hợp lỗi
            fail_record = {
                "query_id": q_id,
                "question": q_text,
                "true_answer": t_ans,
                "true_unit": t_unit,
                "pred_answer": "ERROR",
                "pred_unit": "",
                "latency_seconds": round(latency, 3),
                "error_message": err_msg,
                "p1_correct": False,
                "answer_correct": False,
                "unit_correct": False,
                "has_explanation": False,
                "reason": err_msg
            }
            results_detail.append(fail_record)
            continue

        total_answered += 1
        
        # 3. Tiến hành Chấm điểm P1
        grade = grade_prediction(pred_obj, t_ans, t_unit, expected_query_id=q_id, tolerance=0.05)
        
        is_p1 = grade["p1_correct"]
        if is_p1:
            total_correct += 1
            status_icon = "[PASS]"
        else:
            status_icon = "[FAIL]"

        p_ans = pred_obj.get("answer", "")
        p_u = pred_obj.get("unit", "")
        print(f"{status_icon} ({latency:.2f}s) | Pred: {p_ans} {p_u} | True: {t_ans} {t_unit}")

        # Lưu lại để xuất submission
        submission_list.append(pred_obj)

        # Lưu chi tiết bản ghi
        record = {
            "query_id": q_id,
            "question": q_text,
            "true_answer": t_ans,
            "true_unit": t_unit,
            "pred_answer": p_ans,
            "pred_unit": p_u,
            "explanation": pred_obj.get("explanation", ""),
            "latency_seconds": round(latency, 3),
            "p1_correct": grade["p1_correct"],
            "query_id_match": grade.get("query_id_match", True),
            "answer_correct": grade["answer_correct"],
            "unit_correct": grade["unit_correct"],
            "has_explanation": grade["has_explanation"],
            "grade_reason": grade["reason"],
            "parts_detail": grade["parts_detail"],
        }
        results_detail.append(record)

        if delay > 0 and not mock_mode:
            time.sleep(delay)

    # ============================================================
    # 4. TỔNG HỢP VÀ BÁO CÁO THỐNG KÊ
    # ============================================================
    avg_latency = total_latency / total_questions if total_questions > 0 else 0.0
    accuracy = (total_correct / total_questions * 100.0) if total_questions > 0 else 0.0

    summary = {
        "total_questions": total_questions,
        "total_answered": total_answered,
        "total_correct": total_correct,
        "p1_accuracy_percent": round(accuracy, 2),
        "total_timeouts": total_timeouts,
        "total_http_errors": total_http_errors,
        "avg_latency_seconds": round(avg_latency, 2),
        "total_time_seconds": round(total_latency, 2),
    }

    print("\n" + "=" * 70)
    print("[SUMMARY] BANG TONG KET KET QUA DANH GIA (ViPRA 2026 P1)")
    print("=" * 70)
    print(f"  - Tong so cau hoi kiem thu : {total_questions}")
    print(f"  - So cau tra loi thanh cong : {total_answered}/{total_questions}")
    print(f"  - SO CAU DAT DIEM CHUAN P1   : {total_correct}/{total_questions} ({accuracy:.2f}%)")
    print(f"  - So cau bi Timeout (>60s)  : {total_timeouts}")
    print(f"  - So cau loi HTTP / Server  : {total_http_errors}")
    print(f"  - Thoi gian phan hoi TB     : {avg_latency:.2f} giay/cau")
    print("=" * 70)

    # 5. Lưu kết quả ra file JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    full_output = {
        "summary": summary,
        "results": results_detail,
        "failed_cases": [r for r in results_detail if not r["p1_correct"]]
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Đã lưu báo cáo chi tiết vào: {output_json_path}")

    # Lưu submission format chuẩn BTC
    if submission_list:
        os.makedirs(os.path.dirname(submission_json_path), exist_ok=True)
        with open(submission_json_path, "w", encoding="utf-8") as f:
            json.dump(submission_list, f, ensure_ascii=False, indent=2)
        print(f"[+] Đã lưu file submission chuẩn BTC vào: {submission_json_path}")

    return full_output


# ============================================================
# 4. ENTRY POINT DÒNG LỆNH (CLI)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Chấm điểm Live HTTP API ViPRA 2026")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080/predict",
                        help="URL của API dự đoán (/predict, mặc định http://localhost:8080/predict)")
    parser.add_argument("--csv", type=str, default=config.RAW_TEST_CSV,
                        help="Đường dẫn file test_dataset.csv")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout tối đa cho mỗi câu (mặc định 60s)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số câu để test nhanh (VD: --limit 5)")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Nghỉ giữa các request (giây)")
    parser.add_argument("--output", type=str, default="output/test_evaluation_results.json",
                        help="File lưu kết quả chi tiết")
    parser.add_argument("--mock", action="store_true",
                        help="Chạy ở chế độ Mock để kiểm thử logic chấm điểm")

    args = parser.parse_args()

    # Kiểm tra xem có biến môi trường hoặc input API URL không
    api_url = os.environ.get("PREDICT_API_URL", args.api_url)

    run_evaluation(
        api_url=api_url,
        csv_path=args.csv,
        timeout=args.timeout,
        limit=args.limit,
        delay=args.delay,
        output_json_path=args.output,
        mock_mode=args.mock,
    )


if __name__ == "__main__":
    main()
