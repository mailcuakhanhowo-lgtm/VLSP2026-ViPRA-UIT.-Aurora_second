#!/bin/bash
# fix_cuda.sh - Tu dong thiet lap nvcc va lien ket /usr/local/cuda cho RTX 5090

echo "========================================="
echo "   THIET LAP CUDA TOOLKIT & NVCC"
echo "========================================="

# 1. Kiem tra xem he thong da co san /usr/local/cuda-12.* chua
CUDA_SYS_DIR=$(ls -d /usr/local/cuda-12.* 2>/dev/null | head -n 1)

if [ -n "$CUDA_SYS_DIR" ]; then
    echo "[1/3] Phat hien CUDA san co tren he thong tai: $CUDA_SYS_DIR"
    sudo rm -f /usr/local/cuda 2>/dev/null || rm -f /usr/local/cuda 2>/dev/null || true
    sudo ln -s "$CUDA_SYS_DIR" /usr/local/cuda 2>/dev/null || ln -s "$CUDA_SYS_DIR" /usr/local/cuda 2>/dev/null
    echo "-> Da tao lien ket /usr/local/cuda -> $CUDA_SYS_DIR"
else
    echo "[1/3] He thong chua co CUDA. Dang cai dat nvidia-cuda-nvcc vao .venv..."
    uv pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12

    NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null)
    if [ -z "$NVCC_DIR" ]; then
        NVCC_DIR=$(find .venv -type d -path "*/nvidia/cuda_nvcc" 2>/dev/null | head -n 1)
    fi

    if [ -n "$NVCC_DIR" ]; then
        TARGET_PATH=$(realpath "$NVCC_DIR")
        echo "[2/3] Da tim thay nvidia_cuda_nvcc tai: $TARGET_PATH"
        sudo rm -f /usr/local/cuda 2>/dev/null || rm -f /usr/local/cuda 2>/dev/null || true
        sudo ln -s "$TARGET_PATH" /usr/local/cuda 2>/dev/null || ln -s "$TARGET_PATH" /usr/local/cuda 2>/dev/null
        echo "-> Da tao lien ket /usr/local/cuda -> $TARGET_PATH"
    else
        echo "(!) Khong tim thay thu muc cuda_nvcc"
    fi
fi

# 2. Dam bao PATH va CUDA_HOME
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"

# 3. Kiem tra ket qua
echo ""
echo "========================================="
echo "   KIEM TRA NVCC:"
echo "========================================="
if [ -f "/usr/local/cuda/bin/nvcc" ]; then
    /usr/local/cuda/bin/nvcc --version
    echo ""
    echo "-> NVCC DA SAN SANG! FlashInfer se bien dich thanh cong."
elif command -v nvcc &>/dev/null; then
    nvcc --version
    echo ""
    echo "-> NVCC DA CO SAN TRONG HE THONG!"
else
    echo "(!) Chua tim thay nvcc. Kiem tra lai thu muc."
fi
