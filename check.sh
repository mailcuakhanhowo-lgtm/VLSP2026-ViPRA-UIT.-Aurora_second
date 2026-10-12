#!/bin/bash
# check.sh - Kiem tra toan bo he thong noi bo tren VPS

echo "========================================="
echo "   KIEM TRA HE THONG VIPRA 2026 (LOCAL)"
echo "========================================="

echo ""
echo "[1/4] Thong tin GPU va VRAM:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || echo "Khong goi duoc nvidia-smi"

echo ""
echo "[2/4] Kiem tra vLLM Engine (Port 8000)..."
curl -s http://127.0.0.1:8000/v1/models | grep -q "Qwen" && echo "-> vLLM OK: Qwen2.5-7B-Instruct dang hoat dong tot." || echo "-> vLLM CHUA SAN SANG (kiem tra: tail -n 20 vllm.log)"

echo ""
echo "[3/4] Kiem tra API Server (Port 8080)..."
curl -s http://127.0.0.1:8080/health && echo "" || echo "-> API Server CHUA SAN SANG (kiem tra: tail -n 20 server.log)"

echo ""
echo "[4/4] Chay thu 1 bai toan vat ly mau qua API /predict..."
curl -s -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"query_id": "TEST_LOCAL_01", "query": "Một tụ điện có điện dung C = 100 uF được tích điện đến hiệu điện thế U = 20 V. Tính năng lượng điện trường dự trữ trong tụ điện theo đơn vị Joules (J)?"}' \
  | python3 -m json.tool || echo "Loi khi goi /predict"

echo ""
echo "========================================="
echo "   HOAN TAT KIEM TRA NOI BO!"
echo "========================================="
