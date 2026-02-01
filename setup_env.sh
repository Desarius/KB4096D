#!/usr/bin/env bash
# Bootstrap: pip, venv, PyTorch with CUDA 12.8, verify GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="python3"

echo "=== KB4096D Environment Setup ==="
echo "Working directory: $SCRIPT_DIR"

# --- Step 1: Ensure pip is available (user-level) ---
if ! "$PYTHON" -m pip --version &>/dev/null; then
    echo "[1/5] pip not found, installing via get-pip.py --user..."
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$PYTHON" /tmp/get-pip.py --user --break-system-packages
    rm /tmp/get-pip.py
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] pip already available: $($PYTHON -m pip --version)"
fi

# --- Step 2: Create virtual environment ---
# Handle missing ensurepip by creating venv --without-pip then bootstrapping
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/python" ]; then
    rm -rf "$VENV_DIR"
    echo "[2/5] Creating virtual environment in $VENV_DIR..."
    if "$PYTHON" -m venv "$VENV_DIR" 2>/dev/null; then
        echo "  Created with ensurepip"
    else
        echo "  ensurepip not available, creating without pip..."
        "$PYTHON" -m venv --without-pip "$VENV_DIR"
        echo "  Bootstrapping pip into venv..."
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        "$VENV_DIR/bin/python" /tmp/get-pip.py
        rm /tmp/get-pip.py
    fi
else
    echo "[2/5] Virtual environment already exists"
fi

# --- Step 3: Activate and verify pip ---
source "$VENV_DIR/bin/activate"
echo "[3/5] Activated venv"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# Make sure pip is in the venv
if ! python -m pip --version &>/dev/null; then
    echo "  Installing pip into venv..."
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python /tmp/get-pip.py
    rm /tmp/get-pip.py
fi
echo "  pip: $(python -m pip --version)"

# --- Step 4: Install dependencies ---
echo "[4/5] Installing dependencies from requirements.txt..."
pip install --upgrade pip

# Detect CUDA availability to choose the right PyTorch index
if python -c "
import subprocess, sys
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    sys.exit(0 if result.returncode == 0 else 1)
except FileNotFoundError:
    sys.exit(1)
" 2>/dev/null; then
    echo "  NVIDIA GPU detected, installing PyTorch with CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu128 2>/dev/null \
        || pip install torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null \
        || pip install torch
    pip install transformers accelerate safetensors pytest
else
    echo "  No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers accelerate safetensors pytest
fi

# --- Step 5: Verify ---
echo "[5/5] Verifying setup..."
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f'  VRAM: {vram / 1024**3:.1f} GB')
else:
    print('  No CUDA — CPU mode will be used.')
import transformers
print(f'  Transformers version: {transformers.__version__}')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run with:      python run.py"
echo "Test with:     pytest tests/ -v"
