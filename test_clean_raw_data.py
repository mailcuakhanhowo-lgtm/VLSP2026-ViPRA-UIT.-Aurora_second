"""
Bộ kiểm thử chuẩn Pytest cho quy trình tiền xử lý và làm sạch HTML thô (Clean Raw Data).
Xác thực:
1. Tiền xử lý DOM bằng BeautifulSoup giải cứu công thức toán học MathJax từ <script type="math/tex">.
2. Bảo toàn công thức toán khối ($$...$$) và nội dòng ($...$).
3. Ngăn ngừa lỗi thoát ký tự gạch dưới (\\_) trong công thức LaTeX khi qua bộ bóc tách Markdown.
4. Loại bỏ hoàn toàn các thẻ HTML rác, script UI, button và footer.
5. Kiểm thử toàn diện trên dữ liệu thực tế từ bài giảng Feynman (nếu có cache).
"""

import os
import re
import pytest

from src.latex_parser import rescue_math_with_bs4, clean_html_to_markdown_with_latex, clean_html_fallback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def mock_mathjax_html():
    """Fixture cung cấp đoạn HTML giả lập chứa cấu trúc MathJax đặc thù."""
    return r"""
    <div class="article-content">
        <h1>Coulomb's Law and Electrostatics</h1>
        <p>The electrostatic force between two point charges is given by Coulomb's law:</p>
        <div class="display-math-container">
            <script type="math/tex; mode=display">F = \frac{1}{4\pi\epsilon_0} \frac{q_1 q_2}{r^2}</script>
        </div>
        <p>In this equation, <script type="math/tex">\epsilon_0</script> represents the electric constant,
        and <script type="math/tex">r_{12}</script> is the separation vector between charges.</p>
        <div class="ui-junk">
            <button class="btn-share">Share this section</button>
            <script type="text/javascript">console.log("tracking_code_123");</script>
        </div>
    </div>
    """


def test_rescue_math_inline_and_display(mock_mathjax_html):
    """Kiểm tra BeautifulSoup giải cứu chính xác thẻ script math/tex thành ký hiệu LaTeX."""
    sanitized = rescue_math_with_bs4(mock_mathjax_html)

    # Thẻ script math/tex không còn tồn tại trên cây DOM
    assert "math/tex" not in sanitized
    assert '<script type="math/tex"' not in sanitized

    # Công thức display được bọc trong $$...$$
    assert r"$$F = \frac{1}{4\pi\epsilon_0} \frac{q_1 q_2}{r^2}$$" in sanitized

    # Công thức inline được bọc trong $...$
    assert r"$\epsilon_0$" in sanitized
    assert r"$r_{12}$" in sanitized


def test_clean_html_to_markdown_preserves_latex(mock_mathjax_html):
    """Kiểm tra quy trình làm sạch HTML -> Markdown bảo toàn 100% công thức toán học."""
    cleaned = clean_html_to_markdown_with_latex(mock_mathjax_html)

    # 1. Bảo toàn công thức display
    assert r"$$F = \frac{1}{4\pi\epsilon_0} \frac{q_1 q_2}{r^2}$$" in cleaned

    # 2. Bảo toàn công thức inline
    assert r"$\epsilon_0$" in cleaned
    assert r"$r_{12}$" in cleaned

    # 3. Không bị lỗi biến đổi dấu gạch dưới thành \_
    assert r"\epsilon\_0" not in cleaned
    assert r"r\_{12}" not in cleaned

    # 4. Loại bỏ sạch sẽ mã độc / script rác UI
    assert "tracking_code_123" not in cleaned
    assert "Share this section" not in cleaned
    assert "<script" not in cleaned
    assert "<button" not in cleaned


@pytest.mark.parametrize("tex_expr,is_display", [
    (r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}", True),
    (r"\nabla \cdot \mathbf{B} = 0", True),
    (r"E = mc^2", False),
    (r"\hbar = \frac{h}{2\pi}", False),
])
def test_parameterized_math_expressions(tex_expr, is_display):
    """Kiểm tra khả năng bảo toàn với nhiều dạng công thức toán học khác nhau."""
    if is_display:
        html = f'<div><p>Maxwell Equation:</p><script type="math/tex; mode=display">{tex_expr}</script></div>'
        expected = f"$${tex_expr}$$"
    else:
        html = f'<div><p>Formula is <script type="math/tex">{tex_expr}</script> as defined.</p></div>'
        expected = f"${tex_expr}$"

    result = clean_html_to_markdown_with_latex(html)
    assert expected in result


def test_fallback_cleaner_direct():
    """Kiểm tra fallback cleaner khi HTML chứa các thẻ phân đoạn và công thức delimiters."""
    html = r"""
    <h3>Electromagnetic Energy</h3>
    <p>Energy density is given by:</p>
    \[ u = \frac{1}{2} \left( \epsilon_0 E^2 + \frac{1}{\mu_0} B^2 \right) \]
    <p>where \( E \) and \( B \) are field magnitudes.</p>
    """
    sanitized = rescue_math_with_bs4(html)
    cleaned = clean_html_fallback(sanitized)

    assert r"$$u = \frac{1}{2} \left( \epsilon_0 E^2 + \frac{1}{\mu_0} B^2 \right)$$" in cleaned
    assert r"$E$" in cleaned
    assert r"$B$" in cleaned
    assert "<h3>" not in cleaned
    assert "<p>" not in cleaned


def test_real_cached_feynman_file():
    """Kiểm thử trên dữ liệu cache thực tế của bài giảng Feynman nếu có."""
    cached_file = os.path.join(PROJECT_ROOT, "data", "raw", "feynman_vol_II", "raw_html", "II_01.html")
    if not os.path.exists(cached_file):
        pytest.skip(f"Chưa có file cache {cached_file}, bỏ qua bài kiểm thử dữ liệu thực tế.")

    with open(cached_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Trích xuất 1 phân đoạn section thực tế
    m_start = re.search(r'id=["\']Ch1-S1["\']', content)
    m_end = re.search(r'id=["\']Ch1-S2["\']', content)
    sample = content[m_start.start():m_end.start()] if (m_start and m_end) else content[:15000]

    cleaned = clean_html_to_markdown_with_latex(sample)

    assert len(cleaned) > 200, "Văn bản trích xuất quá ngắn, có thể bị lỗi làm sạch."
    assert "<script" not in cleaned, "Vẫn còn thẻ script trong văn bản sau bóc tách."
    assert "$" in cleaned, "Không tìm thấy ký hiệu công thức toán trong bài giảng vật lý."
