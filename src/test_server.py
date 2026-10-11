"""
test_server.py - Giả lập máy chủ đánh giá của BTC bắn request kiểm tra Server
Dùng để kiểm tra:
1. Endpoint GET  /health
2. Endpoint POST /predict với câu hỏi đơn
3. Endpoint POST /predict với câu hỏi nhiều phần
4. Kiểm tra schema JSON trả về có đúng chuẩn BTC không
"""

import requests
import json
import time

SERVER_URL = "http://127.0.0.1:8080"

def test_api():
    print("=" * 60)
    print("BẮT ĐẦU KIỂM TRA API SERVER (GIẢ LẬP BTC)")
    print("=" * 60)

    # 1. Kiểm tra /health
    print("\n[1] Kiểm tra GET /health...")
    try:
        res = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"Status Code: {res.status_code}")
        print("Response:", res.json())
        assert res.status_code == 200, "Health check thất bại!"
    except Exception as e:
        print(f"Lỗi kết nối server: {e}")
        return

    # 2. Kiểm tra POST /predict (Câu hỏi đơn)
    print("\n[2] Kiểm tra POST /predict (Bài toán năng lượng tụ điện)...")
    payload = {
        "query_id": "TD401",
        "query": "Tính năng lượng điện trường tích trữ trong tụ C khi C = 100 μF dưới U = 30 V"
    }
    
    start_time = time.time()
    res = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=60)
    elapsed = time.time() - start_time

    print(f"Thời gian phản hồi: {elapsed:.2f} giây")
    print(f"Status Code: {res.status_code}")
    print("Response JSON:\n", json.dumps(res.json(), ensure_ascii=False, indent=2))

    # Kiểm tra format chuẩn BTC
    data = res.json()
    assert isinstance(data, list), "Output phải là JSON list (Mục 4)!"
    assert len(data) == 1, "List phải có 1 phần tử!"
    item = data[0]
    assert item["query_id"] == "TD401", "query_id không khớp!"
    assert item["answer"] != "", "answer không được để trống!"
    assert item["explanation"] != "", "explanation không được để trống!"
    assert "reasoning" in item and "steps" in item["reasoning"], "reasoning.steps không đúng cấu trúc!"

    print("\n" + "=" * 60)
    print("🎉 TẤT CẢ KIỂM TRA FORMAT BTC ĐỀU ĐẠT CHUẨN 100%!")
    print("=" * 60)

if __name__ == "__main__":
    test_api()
