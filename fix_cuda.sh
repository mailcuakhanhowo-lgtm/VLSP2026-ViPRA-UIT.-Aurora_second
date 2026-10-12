#!/bin/bash
# fix_cuda.sh - Tich hop CUDA Toolkit / nvcc vao moi truong uv va he thong

echo "========================================="
echo "   THIET LAP NVCC CHO MOI TRUONG UV"
echo "========================================="

# 1. Kiem tra xem he thong da co san /usr/local/cuda-12.* chua
CUDA_SYS_DIR=$(ls -d /usr/local/cuda-12.* 2>/dev/null | head -n 1)

if [ -n "$CUDA_SYS_DIR" ]; then
    echo "[1/3] Phat hien CUDA san co tren he thong tai: $CUDA_SYS_DIR"
    CUDA_TARGET="$CUDA_SYS_DIR"
else
    echo "[1/3] Cai dat nvidia-cuda-nvcc vao moi truong uv..."
    uv pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12

    CUDA_TARGET=$(uv run python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null)
    if [ -z "$CUDA_TARGET" ]; then
        CUDA_TARGET=$(find .venv -type d -path "*/nvidia/cuda_nvcc" 2>/dev/null | head -n 1)
    fi
    echo "[2/3] Da xac dinh thu muc NVCC trong uv: $CUDA_TARGET"
fi

# 2. Tao lien ket /usr/local/cuda neu co quyen
if [ -n "$CUDA_TARGET" ] && [ -d "$CUDA_TARGET" ]; then
    TARGET_ABS=$(realpath "$CUDA_TARGET")
    if [ ! -d "/usr/local/cuda" ] || [ -L "/usr/local/cuda" ]; then
        echo "[3/3] Tao lien ket /usr/local/cuda -> $TARGET_ABS..."
        sudo rm -f /usr/local/cuda 2>/dev/null || rm -f /usr/local/cuda 2>/dev/null || true
        sudo ln -s "$TARGET_ABS" /usr/local/cuda 2>/dev/null || ln -s "$TARGET_ABS" /usr/local/cuda 2>/dev/null || true
    fi
fi

# 3. Kiem tra nvcc
echo ""
echo "========================================="
echo "   KIEM TRA NVCC:"
echo "========================================="
if [ -f "/usr/local/cuda/bin/nvcc" ]; then
    /usr/local/cuda/bin/nvcc --version
    echo ""
    echo "-> NVCC tai /usr/local/cuda da san sang!"
elif [ -n "$CUDA_TARGET" ] && [ -f "$CUDA_TARGET/bin/nvcc" ]; then
    "$CUDA_TARGET/bin/nvcc" --version
    echo ""
    echo "-> NVCC trong uv (.venv) da san sang!"
elif command -v nvcc &>/dev/null; then
    nvcc --version
    echo ""
    echo "-> NVCC trong PATH da san sang!"
else
    echo "(!) Canh bao: Chua tim thay file nvcc."
fi
