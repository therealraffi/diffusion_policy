#!/bin/bash
# Setup script for Safe Diffusion environment
# Follows SETUP.md step-by-step

set -e

# ============================================================================
# PARAMETERS - MODIFY THESE FOR YOUR SYSTEM
# ============================================================================

WORKSPACE="/bigtemp/rhm4nj/safe_diffusion"
ENV_PREFIX="/bigtemp/rhm4nj/envs/safediff"
PYTORCH_CUDA="12.4"

# ============================================================================
# STEP 0: VERIFICATION
# ============================================================================

echo "Setting up Safe Diffusion environment..."
echo "WORKSPACE: $WORKSPACE"
echo "ENV_PREFIX: $ENV_PREFIX"
echo "PYTORCH_CUDA: $PYTORCH_CUDA"
echo ""

if [ ! -d "$WORKSPACE" ]; then
    echo "ERROR: WORKSPACE does not exist: $WORKSPACE"
    exit 1
fi

echo "OK: Workspace valid"
echo ""

# ============================================================================
# CLONE REPOS IF NEEDED
# ============================================================================

if [ ! -d "$WORKSPACE/diffusion_policy" ]; then
    echo "Cloning diffusion_policy..."
    cd "$WORKSPACE"
    git clone https://github.com/cheng-chi/diffusion_policy.git
    cd - > /dev/null
fi

if [ ! -d "$WORKSPACE/robosuite" ]; then
    echo "Cloning robosuite..."
    cd "$WORKSPACE"
    git clone https://github.com/ARISE-Initiative/robosuite.git
    cd - > /dev/null
fi

echo ""

# ============================================================================
# STEP 1: CREATE CONDA ENVIRONMENT
# ============================================================================

echo "STEP 1: Creating conda environment..."

if [ -d "$ENV_PREFIX" ]; then
    echo "Warning: $ENV_PREFIX already exists"
    read -p "Continue and use existing env? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    conda create --prefix "$ENV_PREFIX" python=3.10 -y
fi

echo ""

# ============================================================================
# STEP 2: INSTALL PYTORCH
# ============================================================================

echo "STEP 2: Installing PyTorch..."

conda run -p "$ENV_PREFIX" \
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    pytorch-cuda=$PYTORCH_CUDA -c pytorch -c nvidia -y

echo "Verifying PyTorch..."
conda run -p "$ENV_PREFIX" python -c "import torch; print('PyTorch:', torch.__version__)"

echo ""

# ============================================================================
# STEP 3: FIX MKL COMPATIBILITY
# ============================================================================

echo "STEP 3: Fixing MKL 2025 compatibility..."

conda run -p "$ENV_PREFIX" \
    conda install mkl=2024.0 intel-openmp=2024.0 -c conda-forge -y

echo ""

# ============================================================================
# STEP 4: INSTALL MUJOCO
# ============================================================================

echo "STEP 4: Installing MuJoCo..."

conda run -p "$ENV_PREFIX" pip install mujoco pyopengl

conda run -p "$ENV_PREFIX" python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"

echo ""

# ============================================================================
# STEP 5: SET HEADLESS RENDERING
# ============================================================================

echo "STEP 5: Configuring headless rendering..."

if ! grep -q "MUJOCO_GL" ~/.bashrc; then
    echo "export MUJOCO_GL=egl" >> ~/.bashrc
fi

export MUJOCO_GL=egl

echo "Testing headless rendering..."
conda run -p "$ENV_PREFIX" python -c "
import mujoco, numpy as np
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\".1\"/></body></worldbody></mujoco>')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
mujoco.mj_step(model, data)
renderer.update_scene(data)
pixels = renderer.render()
print('Render shape:', pixels.shape)
"

echo ""

# ============================================================================
# STEP 6: INSTALL ROBOSUITE
# ============================================================================

echo "STEP 6: Installing robosuite..."

conda run -p "$ENV_PREFIX" pip install -e "$WORKSPACE/robosuite"

echo "Applying namespace package fix to robosuite finder..."
ROBOSUITE_FINDER="$ENV_PREFIX/lib/python3.10/site-packages/__editable___robosuite_1_5_2_finder.py"
if [ -f "$ROBOSUITE_FINDER" ]; then
    sed -i 's/sys\.meta_path\.append(_EditableFinder)/sys.meta_path.insert(0, _EditableFinder)/g' "$ROBOSUITE_FINDER"
    echo "Patched robosuite finder"
fi

echo ""

# ============================================================================
# STEP 8: INSTALL DIFFUSION_POLICY
# ============================================================================

echo "STEP 7: Installing diffusion_policy..."

conda run -p "$ENV_PREFIX" pip install -e "$WORKSPACE/diffusion_policy"

echo "Applying namespace package fix to diffusion_policy finder..."
DIFPOL_FINDER="$ENV_PREFIX/lib/python3.10/site-packages/__editable___diffusion_policy_0_0_0_finder.py"
if [ -f "$DIFPOL_FINDER" ]; then
    sed -i 's/sys\.meta_path\.append(_EditableFinder)/sys.meta_path.insert(0, _EditableFinder)/g' "$DIFPOL_FINDER"

    # Add MAPPING entry if not present
    if ! grep -q "diffusion_policy.*$WORKSPACE" "$DIFPOL_FINDER"; then
        sed -i "/^MAPPING.*= {}/c MAPPING: dict[str, str] = {\n    'diffusion_policy': '$WORKSPACE/diffusion_policy/diffusion_policy'\n}" "$DIFPOL_FINDER"
    fi
    echo "Patched diffusion_policy finder"
fi

echo ""

# ============================================================================
# STEP 9: INSTALL PYTHON DEPENDENCIES
# ============================================================================

echo "STEP 8: Installing Python dependencies..."

conda run -p "$ENV_PREFIX" pip install \
    einops \
    "diffusers==0.29.2" \
    hydra-core \
    omegaconf \
    zarr \
    wandb \
    pytorch3d

echo ""

echo "STEP 9: Installing robomimic (no-deps)..."

conda run -p "$ENV_PREFIX" pip install robomimic==0.2.0 --no-deps

echo ""

# ============================================================================
# STEP 11: VERIFY FULL STACK
# ============================================================================

echo "STEP 10: Verifying full stack..."

conda run -p "$ENV_PREFIX" python << 'VERIFY'
import torch
import mujoco
import robosuite as suite
import numpy as np

print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('MuJoCo:', mujoco.__version__)

env = suite.make(
    env_name='Lift',
    robots='Panda',
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names='agentview',
    camera_heights=84,
    camera_widths=84,
)
obs = env.reset()

low, high = env.action_spec
for i in range(10):
    action = np.random.uniform(low, high)
    obs, reward, done, info = env.step(action)

env.close()
print('Full stack verified. All systems go.')
VERIFY

echo ""

# ============================================================================
# DONE
# ============================================================================

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Environment location: $ENV_PREFIX"
echo "Python binary: $ENV_PREFIX/bin/python"
echo ""
echo "To use:"
echo "  $ENV_PREFIX/bin/python train_with_setup.py --config-name=..."
echo ""
echo "Or in SLURM scripts:"
echo "  PYTHON=$ENV_PREFIX/bin/python"
echo "  \$PYTHON train_with_setup.py ..."
echo ""
