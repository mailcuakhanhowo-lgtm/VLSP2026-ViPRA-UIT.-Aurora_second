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
from data_loader import load_train, load_dev; print("  OK data_loader")
from solver import PhysicsSolver, PhysicsPAL; print("  OK solver")
from metric import clean_python_code, execute_code, btc_correctness_metric; print("  OK metric")

# 2. Data loader
print("\n[2] Kiem tra data_loader...")
train = load_train()
dev   = load_dev()
ex = train[0]
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

# 7. Config paths
print("\n[7] Kiem tra duong dan config...")
print(f"  RAW_TRAIN_CSV = {config.RAW_TRAIN_CSV}")
print(f"  RAW_DEV_CSV   = {config.RAW_DEV_CSV}")
print(f"  BASE_MODEL_ID = {config.BASE_MODEL_ID}")
print(f"  LM_API_BASE   = {config.LM_API_BASE}")
print(f"  OPTIMIZED_PROMPT_PATH = {config.OPTIMIZED_PROMPT_PATH}")
print(f"  Train CSV ton tai: {os.path.exists(config.RAW_TRAIN_CSV)}")
print(f"  Dev CSV ton tai  : {os.path.exists(config.RAW_DEV_CSV)}")

# 8. Kiểm tra placeholder protection (không cần download model)
print("\n[8] Kiem tra placeholder protection (khong can model)...")
from translator import _protect_values, _restore_values

test_vi = "Mach noi tiep XL = 22 Ohm, XC = 198 Ohm, U = 110 V. Tang tan so 3 lan."
protected_text, mapping = _protect_values(test_vi)
print(f"  Goc: {test_vi}")
print(f"  Sau protect: {protected_text}")
print(f"  Mapping: {mapping}")
restored = _restore_values(protected_text, mapping)
print(f"  Sau restore: {restored}")
print(f"  Khop voi goc: {restored == test_vi}")

# 9. Kiem tra translator import (khong download model - chi check module load)
print("\n[9] Kiem tra translator module import...")
import translator
print(f"  OK translator module")
print(f"  USE_TRANSLATION = {config.USE_TRANSLATION}")
print(f"  TRANSLATION_MODEL_ID = {config.TRANSLATION_MODEL_ID}")
print(f"  TRANSLATION_DEVICE = {config.TRANSLATION_DEVICE}")
print(f"  TRANSLATION_CACHE_DIR = {config.TRANSLATION_CACHE_DIR}")

print("\n" + "=" * 50)
print("TAT CA KIEM TRA HOAN THANH")
print("=" * 50)
