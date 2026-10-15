"""
Module xử lý và trích xuất LaTeX từ HTML của Feynman Lectures on Physics.
Tích hợp quy trình 2 bước:
1. Dùng BeautifulSoup để giải cứu toàn bộ công thức MathJax trong thẻ <script type="math/tex"> thành $...$ và $$...$$
2. Dùng Trafilatura (nếu có) hoặc bộ bóc tách Markdown để làm sạch triệt để mã rác UI mà bảo toàn 100% công thức toán học.
"""

import os
import re
import html
import logging
from bs4 import BeautifulSoup, NavigableString

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

logger = logging.getLogger(__name__)


def rescue_math_with_bs4(html_content: str) -> str:
    """
    Bước 1: Quét cây DOM bằng BeautifulSoup để giải cứu công thức toán từ các thẻ <script>.
    Biến đổi thẻ <script type="math/tex"> thành text $...$ hoặc $$...$$ trực tiếp trên cây DOM.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Tìm tất cả các thẻ <script> chứa mã toán TeX/LaTeX
    math_scripts = soup.find_all("script", type=re.compile(r"math/tex", re.I))

    for tag in math_scripts:
        tex_code = tag.string or ""
        tex_code = tex_code.strip()
        if not tex_code:
            tag.decompose()
            continue

        script_type = tag.get("type", "")
        # Phân biệt công thức khối (display) và công thức nội dòng (inline)
        # Sử dụng thẻ an toàn <p> và <span> để các engine như Trafilatura không xóa nhầm
        if "mode=display" in script_type.lower():
            p_tag = soup.new_tag("p")
            p_tag.string = f"\n\n$${tex_code}$$\n\n"
            tag.replace_with(p_tag)
        else:
            span_tag = soup.new_tag("span")
            span_tag.string = f" ${tex_code}$ "
            tag.replace_with(span_tag)

    # 2. Xử lý các thẻ công thức TeX dạng delimiters nếu có trong văn bản
    sanitized_html = str(soup)
    sanitized_html = re.sub(r'\\\[(.*?)\\\]', lambda m: f"\n\n$${m.group(1).strip()}$$\n\n", sanitized_html, flags=re.DOTALL)
    sanitized_html = re.sub(r'\\\((.*?)\\\)', lambda m: f" ${m.group(1).strip()}$ ", sanitized_html, flags=re.DOTALL)

    return sanitized_html


def clean_html_fallback(sanitized_html: str) -> str:
    """
    Bộ làm sạch HTML dự phòng (Fallback) khi không có Trafilatura hoặc đoạn HTML quá ngắn:
    Loại bỏ các thẻ rác còn lại nhưng giữ nguyên phân đoạn \n\n và công thức LaTeX.
    """
    text = sanitized_html

    # Loại bỏ các thẻ script thông thường, style, svg, nav, button, footer
    text = re.sub(
        r'<(script|style|noscript|svg|nav|header|footer|button)[^>]*>.*?</\1>',
        ' ',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Đổi các thẻ phân đoạn khối thành dòng mới
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(?:p|div|h[1-6]|li|tr|blockquote)[^>]*>', '\n\n', text, flags=re.IGNORECASE)

    # Xóa toàn bộ các thẻ HTML mở và đóng còn lại
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'^[^\n<]*[a-zA-Z0-9_\-]+=["\'][^"\']*["\'][^>]*>', ' ', text)
    text = re.sub(r'<[a-zA-Z0-9_\-]+[^\n>]*$', ' ', text)

    # Giải mã ký tự thực thể HTML
    text = html.unescape(text)

    # Chuẩn hóa khoảng trắng và dòng trống liên tiếp
    lines = text.splitlines()
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
            prev_blank = False
        else:
            if not prev_blank:
                cleaned_lines.append("")
                prev_blank = True

    return "\n".join(cleaned_lines).strip()


def clean_html_to_markdown_with_latex(html_fragment: str) -> str:
    """
    Hàm giao diện chính được gọi bởi scraper:
    1. BeautifulSoup giải cứu công thức toán từ <script type="math/tex">
    2. Dùng Trafilatura (nếu có) để xuất Markdown chuẩn sạch
    3. Dự phòng bằng clean_html_fallback nếu cần
    """
    if not html_fragment:
        return ""

    # Bước 1: Giải cứu công thức toán học MathJax trên cây DOM
    sanitized_html = rescue_math_with_bs4(html_fragment)

    # Bước 2: Dùng Trafilatura nếu có cài đặt
    if HAS_TRAFILATURA:
        try:
            extracted = trafilatura.extract(
                sanitized_html,
                output_format="markdown",
                include_formatting=True,
                include_links=False,
                include_images=False,
                favor_precision=True
            )
            if extracted and len(extracted.strip()) > 50:
                # Khôi phục các ký tự underscore bị Trafilatura escape thành \_ trong công thức toán LaTeX
                extracted = re.sub(
                    r'(\$\$.*?\$\$|\$.*?\$)',
                    lambda m: m.group(1).replace(r'\_', '_'),
                    extracted,
                    flags=re.DOTALL
                )

                # Kiểm tra an toàn: nếu đoạn HTML ban đầu có công thức display $$ mà Trafilatura làm mất, dùng fallback
                if "$$" in sanitized_html and "$$" not in extracted:
                    logger.debug("Trafilatura làm mất công thức block $$, kích hoạt fallback.")
                    return clean_html_fallback(sanitized_html)

                return extracted.strip()
        except Exception as e:
            logger.debug(f"Trafilatura không xử lý được fragment: {e}")

    # Bước 3: Dự phòng sạch sẽ bằng Fallback parser
    return clean_html_fallback(sanitized_html)
