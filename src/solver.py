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
    1. Đọc đề bài và tóm tắt dữ kiện vào trường 'extraction'.
    2. Viết MÃ PYTHON THUẦN TÚY vào trường 'python_code' để trả lời.

    ===================================================================
    BƯỚC ĐẦU TIÊN: Xác định loại câu hỏi trước khi viết code
    ===================================================================

    [LOẠI A - CÂU TÍNH TOÁN]: Đề bài có chứa số liệu cụ thể (điện áp, điện
    dung, tần số, điện trở...) và yêu cầu tính ra một giá trị số.
    Viet code theo dung CAU TRUC 3 PHAN bat buoc:

        # Phan 1: Khai bao bien va quy doi don vi ve he SI
        # (Moi dai luong phai khai bao duoi dang bien doc lap, KHONG the so gop)
        C_uF = 100
        C = C_uF * 1e-6   # F
        U = 30             # V

        # Phan 2: Viet cong thuc duoi dang comment (truoc khi tinh)
        # W = 0.5 * C * U**2

        # Phan 3: Tinh toan va gan vao bien 'final_answer'
        # (Bai nhieu phan: dung list, VD: final_answer = [val1, val2])
        final_answer = 0.5 * C * U**2
        unit = "J"

    [LOẠI B - CÂU LÝ THUYẾT / ĐỊNH TÍNH]: Đề bài KHÔNG có số liệu cụ thể,
    chỉ hỏi về hiện tượng, chiều biến đổi, điều kiện xảy ra... (VD: "Khi cong
    huong xay ra thi dong dien tang hay giam?", "Tu dien co cho dong mot chieu
    qua khong?").
    KHONG bia ra bien so hay cong thuc gia. Chi can 2 dong:

        # Cau hoi ly thuyet, khong co du kien so
        final_answer = "Tang"   # hoac "Khong", "Giam", "Bang nhau", v.v.
        unit = ""

    ===================================================================
    QUY TAC CHUNG (ap dung cho CA HAI loai)
    ===================================================================
    - Bien cuoi PHAI ten la 'final_answer' (khong doi ten).
    - Bien 'unit' chua don vi ASCII (V, A, ohm, J, uF...). De "" neu khong co.
    - Bai nhieu phan (Loai A): final_answer = [val1, val2], unit = "uC; uJ"
    - KHONG dung print(). KHONG import thu vien ngoai (chi duoc dung math).
    - KHONG viet markdown, KHONG co dau ```python```.
    """

    question: str = dspy.InputField(
        desc="De bai vat ly"
    )

    extraction: str = dspy.OutputField(
        desc=(
            "Tom tat cac du kien da cho va dai luong can tim. "
            "Neu la cau ly thuyet, ghi ro: 'Cau ly thuyet - Hoi ve [chu de].' "
            "Neu la cau tinh toan: 'Cho C = 100 uF, U = 30 V. Tinh W.'"
        )
    )

    python_code: str = dspy.OutputField(
        desc=(
            "Cau tinh toan (Loai A): Ma Python 3 phan bat buoc, "
            "co 'final_answer' (so hoac list) va 'unit'. "
            "Cau ly thuyet (Loai B): Chi 2 dong: "
            "final_answer = 'chuoi ngan gon' va unit = ''. "
            "KHONG co markdown, KHONG co dau ```."
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
