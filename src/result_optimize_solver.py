"""
solver.py - DSPy Signature & Module theo hướng PAL (Program-Aided Language)
LLM chỉ sinh Python code, KHÔNG tự tính số.
Tối giản hóa: DUY NHẤT 1 trường đầu ra 'python_code' để DeepSeek-Math đạt độ chính xác tối đa.
"""
import dspy


# ============================================================
# SIGNATURE: PhysicsRAG_PAL (Program-Aided Language with RAG Context)
# ============================================================
class PhysicsRAG_PAL(dspy.Signature):
    """
    You are an expert in Physics and Python programming following the Program-Aided Language (PAL) approach.
    Read the physics question and the provided RAG context (relevant electromagnetism theory and formulas).
    Write clean, executable Python code to solve the problem numerically.

    Rules:
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

    context: str = dspy.InputField(
        desc="Physics theory, electromagnetism laws, and formulas from RAG."
    )
    question: str = dspy.InputField(
        desc="Physics question to solve."
    )
    python_code: str = dspy.OutputField(
        desc="Clean executable Python code defining final_answer and unit. No markdown."
    )


# Backward-compatibility alias
PhysicsPAL = PhysicsRAG_PAL


# ============================================================
# MODULE: PhysicsSolver (PAL + RAG version)
# ============================================================
class PhysicsSolver(dspy.Module):
    """
    Module giải bài toán vật lý theo hướng PAL kết hợp RAG Context.
    LLM chỉ sinh DUY NHẤT trường 'python_code'.
    Việc tính số thật sự do exec() trong Sandbox đảm nhiệm.
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
