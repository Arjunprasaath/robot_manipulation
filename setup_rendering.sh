#!/bin/bash

# Script to help setup MuJoCo rendering for training

echo "=========================================="
echo "MuJoCo Rendering Setup Helper"
echo "=========================================="
echo ""

# Check Python availability
if ! .venv/bin/python --version &> /dev/null; then
    echo "Error: Python not found in .venv/bin/"
    exit 1
fi

# Check MuJoCo installation
echo "[1/4] Checking MuJoCo installation..."
if .venv/bin/python -c "import mujoco" 2>/dev/null; then
    echo "✓ MuJoCo is installed"
else
    echo "✗ MuJoCo not found!"
    exit 1
fi

# Check for GPU
echo ""
echo "[2/4] Checking for GPU..."
if nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    echo "  Recommendation: Use EGL rendering"
    echo "  Run: export MUJOCO_EGL_DEVICE_ID=-1"
    RECOMMENDED="EGL"
else
    echo "✓ No NVIDIA GPU detected"
    echo "  Recommendation: Use OSMesa (CPU) rendering"
    echo "  Run: export MUJOCO_GL=osmesa"
    RECOMMENDED="OSMESA"
fi

# Test OSMesa
echo ""
echo "[3/4] Testing OSMesa rendering..."
if .venv/bin/python -c "import os; os.environ['MUJOCO_GL']='osmesa'; import mujoco" 2>/dev/null; then
    echo "✓ OSMesa rendering available"
else
    echo "✗ OSMesa rendering not available"
    echo "  Install with: sudo apt-get install libosmesa6-dev"
fi

# Provide recommendations
echo ""
echo "[4/4] Recommendations:"
echo "=========================================="
if [ "$RECOMMENDED" = "EGL" ]; then
    echo "For your system (GPU detected):"
    echo ""
    echo "  export MUJOCO_EGL_DEVICE_ID=-1"
    echo "  ./quick_train.sh"
    echo ""
    echo "Or add to ~/.bashrc:"
    echo "  echo 'export MUJOCO_EGL_DEVICE_ID=-1' >> ~/.bashrc"
else
    echo "For your system (no GPU detected):"
    echo ""
    echo "  export MUJOCO_GL=osmesa"
    echo "  ./quick_train.sh"
    echo ""
    echo "Or add to ~/.bashrc:"
    echo "  echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc"
fi

echo "=========================================="
echo ""
echo "The updated train_vlm_ppo.py will automatically"
echo "fall back to working rendering if the first"
echo "attempt fails."
