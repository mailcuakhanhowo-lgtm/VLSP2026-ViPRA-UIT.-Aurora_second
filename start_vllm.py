"""
start_vllm.py - Tự động đọc config.py và khởi chạy vLLM với đúng BASE_MODEL_ID
"""
import os
import sys
import subprocess

# Thêm src vào path để đọc config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config

model_id = config.BASE_MODEL_ID
print(f"[*] Khởi động vLLM với mô hình từ config.py: {model_id}")

cmd = [
    "vllm", "serve", model_id,
    "--host", "127.0.0.1",
    "--port", "8000",
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "2048",
    "--enforce-eager",
    "--trust-remote-code",
    "--dtype", "float16"
]

env = os.environ.copy()
env["VLLM_USE_V1"] = "0"

try:
    subprocess.run(cmd, env=env)
except KeyboardInterrupt:
    print("\n[*] Đã dừng vLLM.")
