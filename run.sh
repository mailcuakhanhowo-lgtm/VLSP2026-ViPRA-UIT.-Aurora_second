#!/bin/bash
# run.sh - Khoi dong toan bo he thong tu dong cho ViPRA 2026

echo "========================================="
echo "   KHOI DONG HE THONG VIPRA 2026"
echo "========================================="

# 1. Tat cac tien trinh cu neu co
pkill -f vllm
pkill -f uvicorn

# 2. Khoi dong vLLM Engine (Port 8000)
echo "[1/2] Dang khoi dong vLLM Engine (Qwen2.5-7B-Instruct)..."
nohup uv run vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --dtype float16 \
  --max-model-len 6144 > vllm.log 2>&1 &

echo "-> vLLM dang chay ngam (Port 8000)."

# 3. Khoi dong Server API thi dau (Port 8080)
echo "[2/2] Dang khoi dong ViPRA API Server (Port 8080)..."
nohup env PYTHONPATH=. uv run uvicorn src.server:app \
  --host 0.0.0.0 \
  --port 8080 > server.log 2>&1 &

echo "-> Server API dang chay ngam (Port 8080)."
echo ""
echo "========================================="
echo "   KHOI DONG THANH CONG! DU LIEU GHI VAO:"
echo "   - vLLM log   : tail -f vllm.log"
echo "   - Server log : tail -f server.log"
echo "========================================="
