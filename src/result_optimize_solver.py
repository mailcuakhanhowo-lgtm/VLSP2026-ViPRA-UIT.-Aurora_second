"""
solver.py - DSPy Signature & Module theo hướng PAL (Program-Aided Language)
LLM chỉ sinh Python code, KHÔNG tự tính số.
"""
import dspy


# ============================================================
# SIGNATURE: PhysicsRAG_PAL (Program-Aided Language with RAG Context)
# ============================================================
class PhysicsRAG_PAL(dspy.Signature):
    """
    Bạn là chuyên gia Vật lý và lập trình Python. Nhiệm vụ:
    1. Đọc kỹ dữ kiện đề bài 'question' và tài liệu tham khảo 'context'.
    2. Tóm tắt dữ kiện vào trường 'extraction'.
    3. Trình bày cơ sở lý thuyết và công thức đại số tổng quát vào 'explanation_vi'.
       QUY TẮC BẮT BUỘC: 'explanation_vi' TUYỆT ĐỐI KHÔNG CHỨA SỐ LIỆU TÍNH TOÁN HAY KẾT QUẢ SỐ.
       Chỉ giải thích bản chất vật lý, cơ sở lý thuyết đằng sau quyết định tính và công thức đại số.
    4. Viết MÃ PYTHON THUẦN TÚY vào trường 'python_code' để thực hiện tính toán số học từ công thức trên.

    ===================================================================
    QUY TẮC XỬ LÝ CONTEXT (RAG):
    ===================================================================
    - Nếu 'context' chứa định luật hoặc công thức khớp với hiện tượng của 'question', hãy trích dẫn và áp dụng công thức đó.
    - Nếu 'context' là chuỗi thông báo fallback ("Không có tài liệu tham khảo bổ sung...") hoặc không liên quan đến đề bài, HÃY BỎ QUA và tự suy luận theo định luật vật lý tiêu chuẩn.
    - Mọi dữ kiện số liệu (U, I, R, C, L, q...) PHẢI trích xuất 100% từ 'question', TUYỆT ĐỐI KHÔNG lấy số liệu giả định trong 'context'.

    ===================================================================
    BƯỚC ĐẦU TIÊN: Xác định loại câu hỏi
    ===================================================================
    [LOẠI A - CÂU TÍNH TOÁN]: Đề bài có chứa số liệu cụ thể và yêu cầu tính ra giá trị số.
    Viết code theo đúng CẤU TRÚC 3 PHẦN bắt buộc:

        # Phan 1: Khai bao bien va quy doi don vi ve he SI
        # (Moi dai luong khai bao thanh bien doc lap, KHONG the so gop)
        C_uF = 100
        C = C_uF * 1e-6   # F
        U = 30             # V

        # Phan 2: Viet cong thuc duoi dang comment (khop voi explanation_vi)
        # W = 0.5 * C * U**2

        # Phan 3: Tinh toan va gan vao bien 'final_answer'
        # (Bai nhieu phan: dung list, VD: final_answer = [val1, val2])
        final_answer = 0.5 * C * U**2
        unit = "J"

    [LOẠI B - CÂU LÝ THUYẾT / ĐỊNH TÍNH]: Đề bài KHÔNG có số liệu cụ thể, chỉ hỏi về
    hiện tượng, chiều biến đổi, điều kiện xảy ra (VD: "Tang 4 lan", "Co", "Giam", "Bang nhau").
    Chỉ cần 2 dòng:
        # Cau hoi ly thuyet, khong co du kien so
        final_answer = "Tang 4 lan"
        unit = ""

    ===================================================================
    QUY TẮC BẮT BUỘC CHO MÃ PYTHON:
    ===================================================================
    - Biến đáp án cuối cùng BẮT BUỘC tên là 'final_answer'.
    - Biến 'unit' chứa đơn vị chuẩn ASCII (V, A, ohm, J, uF, V/m...). Để "" nếu không có.
    - Góc lượng giác (độ): Sử dụng trực tiếp các hàm lượng giác theo ĐỘ đã có sẵn:
      cos_deg(alpha), sin_deg(alpha), tan_deg(alpha) (VD: cos_deg(120)). KHÔNG dùng math.cos(120).
    - Hợp lực / điện trường vectơ: Hai vectơ cùng phương ngược chiều ở trung điểm phải trừ độ lớn: abs(E1 - E2).
    - Hằng số vật lý có sẵn trong môi trường: k = 9e9, eps0 = 8.854e-12, mu0 = 4*pi*1e-7, g = 9.8, pi = math.pi.
    - KHÔNG dùng print(). KHÔNG import thư viện ngoài.
    - KHÔNG viết markdown, KHÔNG có dấu ```python```.
    """

    context: str = dspy.InputField(
        desc="Tai lieu ly thuyet vat ly va cong thuc lien quan tu RAG (hoac chuoi fallback neu khong co)."
    )

    question: str = dspy.InputField(
        desc="De bai vat ly can giai."
    )

    extraction: str = dspy.OutputField(
        desc=(
            "Tom tat cac du kien da cho tu question va dai luong can tim. "
            "Neu la cau ly thuyet, ghi ro de tai can tra loi."
        )
    )

    explanation_vi: str = dspy.OutputField(
        desc=(
            "Loi giai thich tieng Viet ve co so ly thuyet, hien tuong va cong thuc dai so tong quat. "
            "TUYET DOI KHONG DUA CAC CON SO HAY PHEP TINH SO HOC VAO DAY."
        )
    )

    python_code: str = dspy.OutputField(
        desc=(
            "Cau tinh toan (Loai A): Ma Python 3 phan bat buoc tinh toan so hoc theo cong thuc dai so "
            "da neu o explanation_vi, gan vao final_answer va unit. "
            "Cau ly thuyet (Loai B): Chi 2 dong: final_answer = 'chuoi ngan gon' va unit = ''. "
            "KHONG markdown, KHONG co dau ```."
        )
    )


# Backward-compatibility alias
PhysicsPAL = PhysicsRAG_PAL


# ============================================================
# MODULE: PhysicsSolver (PAL + RAG version)
# ============================================================
class PhysicsSolver(dspy.Module):
    """
    Module giải bài toán vật lý theo hướng PAL kết hợp RAG Context.
    LLM sinh extraction + explanation_vi (lý thuyết không số) + python_code (tính số).
    Việc tính số thật sự do exec() đảm nhiệm (trong metric và wrapper).
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(PhysicsRAG_PAL)

    def forward(self, question: str, context: str = None):
        if not context:
            import config
            context = getattr(
                config,
                "RAG_FALLBACK_TEXT",
                "Không có tài liệu tham khảo bổ sung. Hãy sử dụng các định luật vật lý tiêu chuẩn."
            )
        return self.predictor(context=context, question=question)
