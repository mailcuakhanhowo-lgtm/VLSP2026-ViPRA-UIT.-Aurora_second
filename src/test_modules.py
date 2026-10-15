"""Test script - kiem tra cac module ma khong can vLLM"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.stdout.reconfigure(encoding='utf-8')

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
while True:
    pass
"""
val_inf, unit_inf = execute_code(infinite_code, timeout=0.5)
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

# 12. Kiem tra translator import
print("\n[12] Kiem tra translator module import...")
import result_optimize_translator as translator
print(f"  OK translator module")
print(f"  USE_TRANSLATION = {config.USE_TRANSLATION}")

print("\n" + "=" * 50)
print("TAT CA KIEM TRA HOAN THANH - HE THONG SAN SANG")
print("=" * 50)
