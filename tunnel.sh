#!/bin/bash
# tunnel.sh - Tu dong cai dat va ket noi Cloudflare Tunnel

echo "========================================="
echo "   KET NOI CLOUDFLARE TUNNEL (PUBLIC)"
echo "========================================="

# 1. Kiem tra va tai cloudflared neu chua co
if ! command -v cloudflared &> /dev/null; then
    echo "[1/3] Chua tim thay cloudflared. Dang tu dong tai ve..."
    curl -L --progress-bar https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
    chmod +x cloudflared
    if [ "$EUID" -eq 0 ]; then
        mv cloudflared /usr/local/bin/
        CMD="cloudflared"
    else
        CMD="./cloudflared"
    fi
    echo "-> Da tai cloudflared thanh cong."
else
    CMD="cloudflared"
    echo "[1/3] cloudflared da co san tren he thong."
fi

# 2. Lay Token
TOKEN="$1"
if [ -z "$TOKEN" ]; then
    echo ""
    echo "(!) BAN CHUA TRUYEN TOKEN CLOUDFLARE."
    echo "Cach dung: bash tunnel.sh <TOKEN_CUA_BAN>"
    echo "Hoac nhap Token cua ban tai day (lay tu dashboard Cloudflare Tunnel):"
    read -r TOKEN
fi

if [ -z "$TOKEN" ]; then
    echo "Khong co token. Dung lai."
    exit 1
fi

# 3. Chay Tunnel
echo "[2/3] Tat tunnel cu neu co..."
pkill -f cloudflared 2>/dev/null || true

echo "[3/3] Dang khoi dong Tunnel voi Token..."
nohup $CMD tunnel run --token "$TOKEN" > tunnel.log 2>&1 &

sleep 3
echo "-> Tunnel dang chay ngam! 10 dong log moi nhat:"
echo "-----------------------------------------"
tail -n 10 tunnel.log
echo "-----------------------------------------"
echo "Xem chi tiet log bang lenh: tail -f tunnel.log"
