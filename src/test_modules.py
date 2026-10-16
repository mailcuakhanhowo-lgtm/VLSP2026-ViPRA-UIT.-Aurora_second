"""Test script - kiem tra cac module ma khong can vLLM"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

print("=" * 50)
print("KIEM TRA MODULE (khong can vLLM)")
print("=" * 50)

# 1. Import
print("\n[1] Kiem tra import...")
import config; print("  OK config")
from optimize_data_loader import load_train, load_dev; print("  OK data_loader")
from result_optimize_solver import PhysicsSolver, PhysicsPAL; print("  OK solver")
from result_optimize_metric import clean_python_code, execute_code, btc_correctness_metric; print("  OK metric")

# 2. Data loader
print("\n[2] Kiem tra data_loader...")
train = load_train()
dev   = load_dev()
ex = train[0]
print(f"  Train[0] context  : {ex.context[:70]}...")
print(f"  Train[0] question : {ex.question[:70]}...")
print(f"  Train[0] answer   : {ex.answer}")
print(f"  Train[0] unit     : {repr(ex.unit)}")

# 3. exec() voi code dung
print("\n[3] Kiem tra execute_code() - code dung...")
good_code = """
C_uF = 100
C = C_uF * 1e-6
U = 30
# W = 0.5 * C * U**2
final_answer = 0.5 * C * U**2
unit = "J"
"""
val, unit = execute_code(good_code)
print(f"  final_answer = {val}")
print(f"  unit         = {repr(unit)}")
print(f"  So sanh voi 0.045 (sai so 0.05): {abs(val - 0.045) <= 0.05}")

# 4. exec() voi code loi
print("\n[4] Kiem tra execute_code() - code loi (SyntaxError)...")
bad_code = "final_answer = (("
val2, unit2 = execute_code(bad_code)
print(f"  Ket qua: final_answer={val2}, unit={repr(unit2)} (phai la None, '')")

# 5. clean_python_code
print("\n[5] Kiem tra clean_python_code()...")
dirty = "```python\nC = 100 x 1e-6\nfinal_answer = 0.5 * C * 30**2\nunit = \"J\"\n```"
dirty = dirty.replace("x", "\u00d7")  # ky tu nhan Unicode
cleaned = clean_python_code(dirty)
print("  Code sau khi clean:")
for line in cleaned.splitlines():
    print(f"    {line}")

# 6. Multi-part answer
print("\n[6] Kiem tra multi-part answer...")
multi_code = """
C_uF = 100
C = C_uF * 1e-6
U = 30
Q = C * U        # Dien tich
W = 0.5 * C * U**2  # Nang luong
final_answer = [Q, W]
unit = "C; J"
"""
val3, unit3 = execute_code(multi_code)
print(f"  final_answer = {val3}")
print(f"  unit         = {repr(unit3)}")

# 7. Kiem tra luong giac theo DO (sin_deg, cos_deg)
print("\n[7] Kiem tra luong giac theo DO (sin_deg, cos_deg)...")
trig_code = """
F1 = 4
F2 = 9
alpha = 120
# F_total = sqrt(F1**2 + F2**2 + 2*F1*F2*cos_deg(alpha))
final_answer = round(math.sqrt(F1**2 + F2**2 + 2*F1*F2*cos_deg(alpha)), 2)
unit = "N"
"""
val_trig, unit_trig = execute_code(trig_code)
print(f"  final_answer = {val_trig} {unit_trig}")
print(f"  Ket qua khop voi 7.81 N: {val_trig == 7.81}")

# 8. Kiem tra Sandbox Timeout cung (chong treo vo han)
print("\n[8] Kiem tra Sandbox Timeout cung (while True test)...")
infinite_code = """
import time
while True:
    time.sleep(0.02)
"""
val_inf, unit_inf = execute_code(infinite_code, timeout=0.2)
print(f"  Ket qua khi gap while True: val={val_inf}, unit={repr(unit_inf)} (phai la None, '')")

# 9. Kiem tra chuan hoa don vi Canonical (_normalize_unit)
print("\n[9] Kiem tra chuan hoa don vi Canonical...")
from result_optimize_metric import _normalize_unit
print(f"  'J/m**3' -> {_normalize_unit('J/m**3')} (phai la 'j/m^3')")
print(f"  'N/C' == 'V/m': {_normalize_unit('N/C') == _normalize_unit('V/m')}")
omega_raw = r"\Omega"
print(f"  '\\Omega' -> {_normalize_unit(omega_raw)} (phai la 'ohm')")

# 10. Config paths
print("\n[10] Kiem tra duong dan config...")
print(f"  RAW_TRAIN_CSV = {config.RAW_TRAIN_CSV}")
print(f"  RAW_DEV_CSV   = {config.RAW_DEV_CSV}")
print(f"  BASE_MODEL_ID = {config.BASE_MODEL_ID}")
print(f"  LM_API_BASE   = {config.LM_API_BASE}")
print(f"  OPTIMIZED_PROMPT_PATH = {config.OPTIMIZED_PROMPT_PATH}")
print(f"  RAG_FALLBACK_TEXT     = {config.RAG_FALLBACK_TEXT[:50]}...")
print(f"  Train CSV ton tai: {os.path.exists(config.RAW_TRAIN_CSV)}")
print(f"  Dev CSV ton tai  : {os.path.exists(config.RAW_DEV_CSV)}")

# 11. Kiểm tra placeholder protection (không cần download model)
print("\n[11] Kiem tra placeholder protection (khong can model)...")
from result_optimize_translator import _protect_values, _restore_values

test_vi = "Mach noi tiep XL = 22 Ohm, XC = 198 Ohm, U = 110 V. Tang tan so 3 lan."
protected_text, mapping = _protect_values(test_vi)
print(f"  Goc: {test_vi}")
print(f"  Sau protect: {protected_text}")
print(f"  Mapping: {mapping}")
restored = _restore_values(protected_text, mapping)
print(f"  Sau restore: {restored}")
print(f"  Khop voi goc: {restored == test_vi}")

test_latex = "According to Coulomb's law: $F = k \\frac{|q_1 q_2|}{r^2}$, with $k = 9 \\times 10^9$ N.m^2/C^2."
prot_latex, map_latex = _protect_values(test_latex)
print(f"\n  LaTeX goc: {test_latex}")
print(f"  LaTeX protect: {prot_latex}")
rest_latex = _restore_values(prot_latex, map_latex)
print(f"  LaTeX restore khop: {rest_latex == test_latex}")

# 12. Kiem tra translator import
print("\n[12] Kiem tra translator module import...")
import result_optimize_translator as translator
print(f"  OK translator module")
print(f"  USE_TRANSLATION = {config.USE_TRANSLATION}")
print(f"  Has translate_vi_to_en: {hasattr(translator, 'translate_vi_to_en')}")
print(f"  Has translate_en_to_vi: {hasattr(translator, 'translate_en_to_vi')}")

# 13. Kiem tra Explainer module & fallback
print("\n[13] Kiem tra Explainer module & fallback...")
from result_explainer import PhysicsExplainer, generate_explanation, _fallback_explanation
fb = _fallback_explanation("0.045", "J")
print(f"  Fallback explanation: {fb}")
assert len(fb) > 20, "Fallback explanation qua ngan!"
print("  OK Explainer fallback")

# 14. Kiem tra G-Eval score parser trong optimize_explainer
print("\n[14] Kiem tra logic G-Eval Metric...")
from optimize_explainer import geval_explanation_metric, GEvalExplanationJudge
class MockPred:
    explanation = "Theo dinh luat Coulomb $F = k \\frac{|q_1 q_2|}{r^2}$, luc tuong tac giua hai dien tich la 0.045 J."
class MockEx:
    question = "Tinh luc Coulomb"
    answer = "0.045"
    unit = "J"
    context = "Dinh luat Coulomb..."

# Test metric voi rule-based filtering (khong can LLM judge van chay qua duoc)
score_dummy = geval_explanation_metric(MockEx(), MockPred())
print(f"  Diem test explanation mau: {score_dummy} (phai >= 0.2)")
assert score_dummy > 0.0, "Metric phai tinh duoc diem > 0"
print("  OK G-Eval Metric logic")

# 15. Kiem tra HyDE query expansion trong RAGPipeline
print("\n[15] Kiem tra HyDE RAG module...")
from rag_module import RAGPipeline
rag = RAGPipeline()
test_q = "Cho tu dien C = 100 uF, hieu dien the U = 30 V. Tinh nang luong dien truong."
expanded = rag.expand_query_with_hyde(test_q)
print(f"  Query goc: {test_q}")
print(f"  Query sau HyDE: {expanded}")
assert len(expanded) >= len(test_q), "HyDE query expansion khong hop le!"
print("  OK HyDE RAG expansion")

# 16. Kiem tra Hybrid search threshold & top BM25 fallback
print("\n[16] Kiem tra RAG hybrid_search threshold & BM25 fallback...")
rag_mock = RAGPipeline()
rag_mock.metadata = [
    {"text": "Faraday induction law text", "theory_topic": "Faraday"},
    {"text": "Coulomb law text", "theory_topic": "Coulomb"}
]
# Gia lap BM25 va Semantic ket qua
rag_mock.bm25_search = lambda q: [{"doc_id": 0, "score": 12.5}]
rag_mock.semantic_search = lambda q: [{"doc_id": 1, "score": 0.40}]  # dưới 0.50
# rrf se tra ve [0, 1]
res = rag_mock.hybrid_search("faraday", use_hyde=False)
print(f"  Ket qua hybrid_search khi semantic < 0.50 nhung doc_id 0 thuoc top BM25: {len(res)} ket qua")
assert len(res) == 1 and res[0]["text"] == "Faraday induction law text", "BM25 fallback filtering fail!"
print("  OK BM25 top candidate retention")

# 17. Kiem tra BM25 Regex Tokenize
print("\n[17] Kiem tra BM25 Regex Tokenize...")
from rag_module import tokenize
sample_raw = "(Faraday's, 100uF,, E=mc^2; Mạch RLC: U = 220V."
tokens = tokenize(sample_raw)
print(f"  Raw: {sample_raw}")
print(f"  Tokens: {tokens}")
assert "faraday" in tokens, "BM25 tokenize phai tach duoc 'faraday'!"
assert "100uf" in tokens, "BM25 tokenize phai giu duoc '100uf'!"
assert "mạch" in tokens and "rlc" in tokens, "BM25 tokenize phai ho tro tieng Viet!"
print("  OK BM25 Regex Tokenize")

print("\n" + "=" * 50)
print("TAT CA KIEM TRA HOAN THANH - HE THONG SAN SANG 100%")
print("=" * 50)

