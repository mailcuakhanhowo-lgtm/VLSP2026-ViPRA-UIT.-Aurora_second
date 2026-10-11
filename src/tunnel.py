"""
tunnel.py - Tự động tạo Cloudflare Tunnel công khai kết nối vào cổng 8000
Sử dụng: uv run python src/tunnel.py
"""

import time
from pycloudflared import try_cloudflare

PORT = 8000

print("=" * 60)
print(f"Đang khởi tạo Cloudflare Tunnel cho cổng {PORT}...")
print("Vui lòng đợi vài giây...")
print("=" * 60)

try:
    # try_cloudflare sẽ tự động tải binary và mở tunnel
    tunnel = try_cloudflare(port=PORT)
    public_url = tunnel.tunnel
    
    print("\n" + "=" * 60)
    print("🎉 CLOUDFLARE TUNNEL ĐÃ SẴN SÀNG HOẠT ĐỘNG!")
    print(f"👉 Đường link HTTPS công khai của anh:")
    print(f"   {public_url}")
    print(f"\n👉 Đường link test kiểm tra sức khỏe:")
    print(f"   {public_url}/health")
    print(f"\n👉 Endpoint dự đoán của BTC:")
    print(f"   {public_url}/predict")
    print("=" * 60)
    print("\n(Giữ nguyên cửa sổ này để duy trì kết nối tunnel. Bấm Ctrl + C để dừng)\n")
    
    # Giữ tunnel chạy liên tục
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nĐã tắt tunnel.")
except Exception as e:
    print(f"\nLỗi khi khởi chạy tunnel: {e}")
