#!/bin/bash

# Script to fix MuJoCo rendering issues
echo "=========================================="
echo "MuJoCo Rendering Fix Script"
echo "=========================================="
echo ""

echo "[1/2] Installing Python OpenGL packages..."
.venv/bin/pip install PyOpenGL PyOpenGL-accelerate

if [ $? -eq 0 ]; then
    echo "✓ Python packages installed successfully"
else
    echo "✗ Failed to install Python packages"
    exit 1
fi

echo ""
echo "[2/2] Testing rendering setup..."

# Test if OSMesa works
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

python_test=$(cat <<'EOF'
import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

try:
    import mujoco
    print("✓ MuJoCo import successful with OSMesa")
except Exception as e:
    print(f"✗ MuJoCo import failed: {e}")
    exit(1)
EOF
)

.venv/bin/python -c "$python_test"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Success! Rendering is now configured."
    echo "You can run training with:"
    echo "  ./quick_train.sh"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "OSMesa is not available on this system."
    echo ""
    echo "You may need to install system libraries:"
    echo "  sudo apt-get install libosmesa6-dev libgl1-mesa-glx"
    echo ""
    echo "If you don't have sudo access, you may need to:"
    echo "1. Ask your system administrator to install OSMesa"
    echo "2. Use a different rendering backend"
    echo "=========================================="
fi
