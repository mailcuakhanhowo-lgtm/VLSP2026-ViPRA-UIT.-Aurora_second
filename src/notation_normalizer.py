"""
Module chuẩn hóa công thức toán học và ký hiệu vật lý (Notation Normalizer)
Chuyển đổi 100% macro TeX nội bộ của Feynman Lectures (FLP Macros) sang LaTeX chuẩn quốc tế
và đồng bộ với bảng quy chuẩn ký hiệu của cuộc thi (Submission/notation_mapping.csv).
"""

import os
import re
import sys
import glob
import json
import logging
import argparse

# Nạp cấu hình từ bảng điều khiển trung tâm config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NORMALIZER_INPUT_DIR, NORMALIZER_OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Bảng tra cứu trực tiếp các macro hằng số và đại lượng vật lý đặc thù
EXACT_REPLACEMENTS = {
    # 1. Hằng số vật lý & đơn vị
    r"\epsO": r"\epsilon_0",              # Hằng số điện môi chân không (khớp 'ε' trong notation_mapping)
    r"\emf": r"\mathcal{E}",              # Suất điện động (emf)
    r"\voltage": r"V",                     # Điện áp (Voltage)
    r"\selfInd": r"L",                     # Hệ số tự cảm L (trong Z_L = \omega L)
    r"\mutualInd": r"M",                   # Hệ số hỗ cảm M
    r"\lambdabar": r"\bar{\lambda}",       # Bước sóng rút gọn (\lambda / 2\pi)
    r"\ReynoldsR": r"\mathcal{R}",         # Số Reynolds
    r"\Rdot": r".",                        # Dấu chấm thập phân cũ (ví dụ 2\Rdot0 -> 2.0)
    r"\frakz": r"z",                       # Biến phức z
    r"\bendingMom": r"M_b",                # Mômen uốn

    # 2. Toán tử véc-tơ đơn giản
    r"\FLPnabla": r"\nabla",
    r"\Fignabla": r"\nabla",
    r"\fournabla": r"\nabla_\mu",
    r"\FLPzero": r"\mathbf{0}",
    r"\FLPzeroi": r"\mathbf{0}",
    r"\grad": r"\nabla",
    r"\rightarrowding": r"\rightarrow",

    # 3. Ký tự Hy Lạp dạng véc-tơ
    r"\FLPomega": r"\boldsymbol{\omega}",  # Véc-tơ vận tốc góc
    r"\FLPOmega": r"\boldsymbol{\Omega}",
    r"\FLPmu": r"\boldsymbol{\mu}",        # Véc-tơ mômen từ
    r"\FLPdelta": r"\boldsymbol{\delta}",
    r"\FLPtau": r"\boldsymbol{\tau}",      # Véc-tơ mômen lực
    r"\Figmu": r"\boldsymbol{\mu}",
    r"\Figtau": r"\boldsymbol{\tau}",

    # 4. Ký hiệu hiển thị & dấu ba chấm
    r"\dotsb": r"\cdots",
    r"\dotsm": r"\cdots",
    r"\stared": r"^*",
}


def normalize_feynman_latex(text: str) -> str:
    r"""
    Chuẩn hóa toàn bộ các công thức toán học và macro đặc thù của Feynman trong văn bản:
    - Chuyển macro véc-tơ (\FLPE, \FLPB...) sang \mathbf{E}, \mathbf{B}...
    - Chuyển toán tử đạo hàm (\ddp, \ddt) sang \frac{\partial ...}{\partial ...}
    - Chuyển toán tử vi tích phân (\FLPdiv, \FLPcurl, \FLPgrad) sang \nabla \cdot, \nabla \times
    - Chuyển hằng số (\epsO) sang \epsilon_0
    """
    if not text:
        return ""

    result = text

    # ------------------------------------------------------------------
    # 1. Thay thế các macro chính xác theo từ điển
    # ------------------------------------------------------------------
    for old_macro, new_tex in EXACT_REPLACEMENTS.items():
        pattern = re.escape(old_macro) + r"(?![a-zA-Z])"
        # Dùng lambda để tránh re.sub parse nhầm backslash như \e thành bad escape
        result = re.sub(pattern, lambda m, rep=new_tex: rep, result)

    # ------------------------------------------------------------------
    # 2. Chuẩn hóa toán tử đạo hàm riêng \ddp{A}{B} và \ddpl{A}{B}
    # ------------------------------------------------------------------
    # Trường hợp \ddp{A}{B} -> \frac{\partial A}{\partial B}
    result = re.sub(
        r'\\ddp\{([^}]*)\}\{([^}]+)\}',
        lambda m: f"\\frac{{\\partial {m.group(1)}}}{{\\partial {m.group(2)}}}" if m.group(1).strip() else f"\\frac{{\\partial}}{{\\partial {m.group(2)}}}",
        result
    )
    # Trường hợp \ddpl{A}{B} -> \partial A / \partial B
    result = re.sub(
        r'\\ddpl\{([^}]*)\}\{([^}]+)\}',
        lambda m: f"\\partial {m.group(1)} / \\partial {m.group(2)}" if m.group(1).strip() else f"\\partial / \\partial {m.group(2)}",
        result
    )

    # ------------------------------------------------------------------
    # 3. Chuẩn hóa toán tử đạo hàm thời gian toàn phần \ddt{A}{B}
    # ------------------------------------------------------------------
    result = re.sub(
        r'\\ddt\{([^}]*)\}\{([^}]+)\}',
        lambda m: f"\\frac{{d {m.group(1)}}}{{d {m.group(2)}}}" if m.group(1).strip() else f"\\frac{{d}}{{d {m.group(2)}}}",
        result
    )

    # ------------------------------------------------------------------
    # 4. Chuẩn hóa toán tử vi tích phân véc-tơ: Div, Curl, Grad
    # ------------------------------------------------------------------
    # \FLPdiv{A} -> \nabla \cdot A
    result = re.sub(r'\\FLPdiv\{([^}]+)\}', lambda m: f"\\nabla \\cdot {m.group(1)}", result)
    result = re.sub(r'\\ndiv\s+([\\a-zA-Z0-9]+)', lambda m: f"\\nabla \\cdot {m.group(1)}", result)

    # \FLPcurl{A} -> \nabla \times A
    result = re.sub(r'\\FLPcurl\{([^}]+)\}', lambda m: f"\\nabla \\times {m.group(1)}", result)
    result = re.sub(r'\\curl\s+([\\a-zA-Z0-9]+)', lambda m: f"\\nabla \\times {m.group(1)}", result)

    # \FLPgrad{T} -> \nabla T
    result = re.sub(r'\\FLPgrad\{([^}]+)\}', lambda m: f"\\nabla {m.group(1)}", result)

    # ------------------------------------------------------------------
    # 5. Chuẩn hóa tất cả các véc-tơ \FLP<X> và \Fig<X> sang \mathbf{X}
    # (Bao gồm \FLPE, \FLPB, \FLPA, \FLPv, \FLPF, \FLPj, \FLPP, \FLPs, \FLPr...)
    # ------------------------------------------------------------------
    result = re.sub(r'\\FLP([a-zA-Z])(?![a-zA-Z])', lambda m: f"\\mathbf{{{m.group(1)}}}", result)
    result = re.sub(r'\\Fig([a-zA-Z])(?![a-zA-Z])', lambda m: f"\\mathbf{{{m.group(1)}}}", result)

    # ------------------------------------------------------------------
    # 6. Chuẩn hóa các hàm hiển thị: \tfrac, \dfrac, \av, \abs
    # ------------------------------------------------------------------
    result = re.sub(r'\\(?:tfrac|dfrac)\{([^}]+)\}\{([^}]+)\}', lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}", result)
    result = re.sub(r'\\av\{([^}]+)\}', lambda m: f"\\langle {m.group(1)} \\rangle", result)
    result = re.sub(r'\\abs\{([^}]+)\}', lambda m: f"|{m.group(1)}|", result)

    # Dọn dẹp ký tự Unicode thừa dạng \unicode{x201C} (dấu ngoặc kép)
    result = re.sub(r'\\unicode\{x201[CDcd]\}', '"', result)

    return result


def normalize_dataset(input_dir: str = None, output_dir: str = None, in_place: bool = False):
    """
    Chuẩn hóa toàn bộ các file JSON trong dataset:
    - Đọc đường dẫn từ bảng điều khiển trung tâm config.py (NORMALIZER_INPUT_DIR / NORMALIZER_OUTPUT_DIR)
    - Chuẩn hóa nội dung trường 'content' và 'theory'
    - Lưu vào thư mục processed (hoặc ghi đè in_place)
    """
    if input_dir is None:
        input_dir = NORMALIZER_INPUT_DIR
    if output_dir is None:
        output_dir = input_dir if in_place else NORMALIZER_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_files:
        logger.warning(f"Không tìm thấy file JSON nào trong: {input_dir}")
        return

    logger.info(f"Bắt đầu chuẩn hóa ký hiệu cho {len(json_files)} file JSON...")
    logger.info(f"Nguồn: {input_dir}")
    logger.info(f"Đích:  {output_dir}")

    total_modified = 0
    for idx, fpath in enumerate(json_files, start=1):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_content = data.get("content", "")
        raw_theory = data.get("theory", "")

        norm_content = normalize_feynman_latex(raw_content)
        norm_theory = normalize_feynman_latex(raw_theory)

        data["content"] = norm_content
        data["theory"] = norm_theory

        out_filepath = os.path.join(output_dir, os.path.basename(fpath))
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        total_modified += 1

    logger.info("=" * 60)
    logger.info("HOÀN TẤT CHUẨN HÓA KÝ HIỆU TOÁN HỌC!")
    logger.info(f"- Đã xử lý và lưu {total_modified} file JSON chuẩn hóa vào:")
    logger.info(f"  {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Chuẩn hóa ký hiệu Feynman LaTeX sang chuẩn quốc tế")
    parser.add_argument("--input-dir", type=str, default=None, help="Thư mục chứa file JSON thô")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục lưu file JSON đã chuẩn hóa")
    parser.add_argument("--in-place", action="store_true", help="Ghi đè trực tiếp lên file đầu vào")
    args = parser.parse_args()

    normalize_dataset(input_dir=args.input_dir, output_dir=args.output_dir, in_place=args.in_place)


if __name__ == "__main__":
    main()
