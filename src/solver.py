"""
solver.py - DSPy Signature & Module theo hướng PAL (Program-Aided Language)
LLM chỉ sinh Python code, KHÔNG tự tính số.
"""
import dspy


# ============================================================
# SIGNATURE: PhysicsPAL
# ============================================================
class PhysicsPAL(dspy.Signature):
    """
    Bạn là chuyên gia Vật lý và lập trình Python. Nhiệm vụ:
    1. Đọc đề bài tiếng Việt và tóm tắt dữ kiện vào trường 'extraction'.
    2. Viết MÃ PYTHON THUẦN TÚY vào trường 'python_code' để tính đáp án.

    [QUY TẮC BẮT BUỘC cho python_code]
    Mã PHẢI có đúng 3 phần theo thứ tự sau, không được bỏ qua:

    # Phần 1: Khai báo biến và quy đổi đơn vị về hệ SI
    # (Mỗi đại lượng phải khai báo dưới dạng biến độc lập, KHÔNG thế số gộp)
    C_uF = 100
    C = C_uF * 1e-6   # F
    U = 30             # V

    # Phần 2: Viết công thức dưới dạng comment (trước khi tính)
    # W = 0.5 * C * U**2

    # Phần 3: Tính toán và gán vào biến 'final_answer'
    # (Bài nhiều phần: dùng list, VD: final_answer = [val1, val2])
    final_answer = 0.5 * C * U**2
    unit = "J"

    Lưu ý quan trọng:
    - Biến cuối PHẢI tên là 'final_answer' (không đổi tên).
    - Biến 'unit' chứa đơn vị ASCII (V, A, ohm, J, uF...). Để "" nếu không có.
    - Bài nhiều phần: final_answer = [val1, val2], unit = "uC; uJ"
    - KHÔNG dùng print(). KHÔNG import thư viện ngoài (chỉ được dùng math).
    - KHÔNG viết markdown, KHÔNG có dấu ```python```.
    """

    question: str = dspy.InputField(
        desc="Đề bài vật lý tiếng Việt"
    )

    extraction: str = dspy.OutputField(
        desc=(
            "Tóm tắt các dữ kiện đã cho và đại lượng cần tìm. "
            "Ví dụ: 'Cho C = 100 μF, U = 30 V. Tính W (năng lượng điện trường).'"
        )
    )

    python_code: str = dspy.OutputField(
        desc=(
            "Mã Python thuần túy theo đúng 3 phần bắt buộc. "
            "Phải có biến 'final_answer' và biến 'unit' ở cuối. "
            "KHÔNG có markdown, KHÔNG có dấu ```."
        )
    )


# ============================================================
# MODULE: PhysicsSolver (PAL version)
# ============================================================
class PhysicsSolver(dspy.Module):
    """
    Module giải bài toán vật lý theo hướng PAL.
    LLM chỉ sinh extraction + python_code.
    Việc tính số thật sự do exec() đảm nhiệm (trong metric và wrapper).
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(PhysicsPAL)

    def forward(self, question: str):
        return self.predictor(question=question)
