#!/bin/bash
# remote.sh - Tao phien ket noi SSH truc tiep bang tmate (Khong can RDP, khong giat lag)

echo "========================================="
echo "   KHOI TAO PHIEN DIEU KHIEN SSH (TMATE)"
echo "========================================="

# 1. Kiem tra va cai dat tmate
if ! command -v tmate &> /dev/null; then
    echo "Dang cai dat tmate..."
    sudo apt update && sudo apt install -y tmate
fi

# 2. Khoi tao phien lam viec ngam
tmate -S /tmp/tmate.sock kill-session 2>/dev/null || true
tmate -S /tmp/tmate.sock new-session -d
tmate -S /tmp/tmate.sock wait tmate-ready

# 3. Lay thong tin lenh SSH ket noi
SSH_CMD=$(tmate -S /tmp/tmate.sock display -p '#{tmate_ssh}')
WEB_CMD=$(tmate -S /tmp/tmate.sock display -p '#{tmate_web}')

echo ""
echo "================================================================="
echo "   LENH SSH DE DIEU KHIEN TU MAY BAN (MO POWERSHELL / CMD):"
echo "================================================================="
echo ""
echo "   $SSH_CMD"
echo ""
echo "================================================================="
echo "   HOAC MO QUA TRINH DUYET WEB (NEU THICH GIAO DIEN WEB):"
echo "   $WEB_CMD"
echo "================================================================="
echo ""
echo "Ban chi can copy dong lenh SSH tren, dan vao PowerShell may tinh cua ban"
echo "va Enter la dieu khien truc tiep may VPS 5090 muot ma 100%!"
