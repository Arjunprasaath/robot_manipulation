# MuJoCo Rendering Fix

## Common Errors

### Error 1: EGL Device ID Error
```
RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and -1 (inclusive), got 0.
```

### Error 2: PyOpenGL OSMesa Error
```
AttributeError: 'NoneType' object has no attribute 'glGetError'
```

These happen when MuJoCo can't initialize rendering, typically in Singularity containers or headless servers.

## Solutions (Try in Order)

### Solution 1: Install Required Packages

First, try installing the necessary Python packages:

```bash
.venv/bin/pip install PyOpenGL PyOpenGL-accelerate
```

Then run:
```bash
./quick_train.sh
```

### Solution 2: Install System OSMesa Libraries

If Solution 1 doesn't work, install system OSMesa libraries:

```bash
# Ubuntu/Debian (requires sudo/admin access)
sudo apt-get update
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglew-dev

# Then try again
./quick_train.sh
```

### Solution 3: Set Environment Variables Manually

Try setting both required environment variables:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
./quick_train.sh
```

## Alternative: Force OSMesa

If you still encounter issues, explicitly set OSMesa before running:

```bash
export MUJOCO_GL=osmesa
./quick_train.sh
```

## Alternative: Use GPU Rendering (Advanced)

If you're on a system with proper GPU/EGL support and want faster rendering:

```bash
export MUJOCO_EGL_DEVICE_ID=0
./quick_train.sh
```

Check available GPUs:
```bash
nvidia-smi
# Then use the GPU ID you want, e.g., export MUJOCO_EGL_DEVICE_ID=1
```

### Solution 4: Install osmesa (If Solution 2 fails)

If OSMesa is not installed:

```bash
# Ubuntu/Debian
sudo apt-get install libosmesa6-dev

# Conda (if using conda)
conda install -c conda-forge osmesa

# Then retry
export MUJOCO_GL=osmesa
./quick_train.sh
```

## Check Your Setup

### Check MuJoCo Installation
```bash
.venv/bin/python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
```

### Check Available Rendering Backends
```bash
.venv/bin/python -c "
import os
os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco
print('OSMesa available')
"
```

## Permanent Fix

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# For headless servers with GPU
export MUJOCO_EGL_DEVICE_ID=0

# For headless servers without GPU (CPU rendering)
export MUJOCO_GL=osmesa
```

Then reload:
```bash
source ~/.bashrc
```

## What Each Backend Does

| Backend | Description | Use Case |
|---------|-------------|----------|
| **EGL** | GPU rendering | Fastest, requires GPU |
| **OSMesa** | CPU rendering | Slower but works everywhere |
| **GLFW** | Window rendering | For visualization (not headless) |

## For Your Case

Since you're on a server (`/home/sar4384/`), you likely want OSMesa:

1. **Quick fix** (one-time):
   ```bash
   MUJOCO_GL=osmesa ./quick_train.sh
   ```

2. **Permanent fix** (add to ~/.bashrc):
   ```bash
   echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Run training**:
   ```bash
   ./quick_train.sh
   ```

## Still Having Issues?

Try running without offscreen rendering (no camera images):

Edit `train_vlm_ppo.py` and change:
```python
has_offscreen_renderer=True,
```
to:
```python
has_offscreen_renderer=False,
use_camera_obs=False,
```

But this will disable vision-based control (you'll need to modify the policy to use state observations instead).
