#!/bin/bash
# test_public.sh - Kiem tra toan bo cac endpoint cong khai cua he thong qua Internet

DOMAIN="${1:-https://projectkhanhlam.site}"

echo "========================================="
echo "   KIEM TRA DOMAIN CONG KHAI: $DOMAIN"
echo "========================================="

echo ""
echo "[1/3] Kiem tra Health Check (/health)..."
curl -s -w "\nHTTP Code: %{http_code} | Thoi gian phan hoi: %{time_total}s\n" "$DOMAIN/health"

echo ""
echo "[2/3] Kiem tra BTC Model Endpoint (/v1/models)..."
curl -s "$DOMAIN/v1/models" | python3 -m json.tool || curl -s "$DOMAIN/v1/models"

echo ""
echo "[3/3] Gui bai toan mau toi BTC Endpoint (/predict)..."
START_TIME=$(date +%s)
curl -s -X POST "$DOMAIN/predict" \
  -H "Content-Type: application/json" \
  -d '{"query_id": "TEST_PUBLIC_01", "query": "Một vật có khối lượng m = 2 kg dao động điều hòa với biên độ A = 0.1 m và tần số góc omega = 10 rad/s. Tính cơ năng toàn phần của vật theo đơn vị Joules (J)?"}' \
  | python3 -m json.tool || echo "Loi goi predict"
END_TIME=$(date +%s)
DIFF=$((END_TIME - START_TIME))

echo ""
echo "========================================="
echo "   THOI GIAN SUY LUAN: ${DIFF}s (Gioi han BTC <= 60s)"
echo "   KIEM TRA HOAN TAT!"
echo "========================================="
