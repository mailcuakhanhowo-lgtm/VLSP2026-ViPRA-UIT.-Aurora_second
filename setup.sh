#!/bin/bash
# setup.sh - Cai dat bo cong cu bien dich C (build-essential, python3-dev) cho Triton / vLLM

echo "========================================="
echo "   CAI DAT BUILD-ESSENTIAL & PYTHON3-DEV"
echo "========================================="

if [ "$EUID" -ne 0 ]; then
    sudo apt update && sudo apt install -y build-essential python3-dev
else
    apt update && apt install -y build-essential python3-dev
fi

echo ""
echo "========================================="
echo "   KIEM TRA GCC THANH CONG:"
echo "========================================="
gcc --version
