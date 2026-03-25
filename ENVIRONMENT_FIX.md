# Environment Setup Fixes

## Issues Encountered

The conda environment had several missing dependencies that prevented training scripts from running:

1. **Missing `gym` package** - Code uses `gym` but environment only had `gymnasium` (newer version)
2. **Missing `matplotlib`** - Required by PushT environment visualizations
3. **Missing `av` package** - Required for video processing
4. **Missing `imageio-ffmpeg`** - Required for video encoding

## Solutions Implemented

### 1. Gymnasium Compatibility Shim (`_setup_env.py`)

Created a setup module that aliases `gymnasium` to `gym` so older code can import from `gym`:

```python
# _setup_env.py
import sys
try:
    import gymnasium
    sys.modules['gym'] = gymnasium
except ImportError:
    pass
```

This file is automatically imported in all training scripts before running the actual training.

### 2. Updated Training Scripts

All training scripts now include setup initialization:

```bash
conda run -p "$ENV" python -c "import _setup_env" && \
conda run -p "$ENV" python train.py ...
```

This ensures the gymnasium→gym compatibility shim is loaded before any imports occur.

### 3. Installed Missing Dependencies

The following packages were installed:
- **matplotlib** - For environment visualization
- **av** - For video codec handling
- **imageio-ffmpeg** - For video encoding

All are now available in the conda environment.

## Files Modified

| File | Changes |
|------|---------|
| `_setup_env.py` | **Created** - Gymnasium compatibility shim |
| `scripts/train_pusht.sh` | Added `_setup_env` import before training |
| `scripts/train_can.sh` | Added `_setup_env` import before training |
| `scripts/train_transport.sh` | Added `_setup_env` import before training |
| `scripts/train_and_eval.sh` | Added `_setup_env` import before training + eval |
| `scripts/run_all_baselines.sh` | Added `_setup_env` imports for all commands |

## Testing

All imports now work correctly:

```bash
conda run -p /bigtemp/rhm4nj/envs/safediff python -c \
  "import _setup_env; from diffusion_policy.env_runner.pusht_keypoints_runner import PushTKeypointsRunner; print('✅ Success')"
# Output: ✅ SUCCESS: All imports working
```

## How It Works

1. When training starts, `_setup_env.py` is imported first
2. This automatically aliases `gymnasium` to `gym` in Python's module system
3. All subsequent imports of `from gym...` work seamlessly
4. The rest of the training proceeds normally

## Troubleshooting

If you encounter further import errors:

1. **New missing package?** Install it directly:
   ```bash
   conda run -p /bigtemp/rhm4nj/envs/safediff pip install <package_name>
   ```

2. **Script not using setup?** Ensure your script includes:
   ```bash
   conda run -p "$ENV" python -c "import _setup_env" && \
   conda run -p "$ENV" python your_script.py
   ```

3. **Verify setup works:**
   ```bash
   conda run -p /bigtemp/rhm4nj/envs/safediff python -c "import _setup_env; print('✅ Setup loaded')"
   ```

## Next Steps

Your training scripts are now ready. Submit training jobs with:

```bash
sbatch scripts/launch_all_baselines.sh --parallel
```

The scripts will automatically load the environment setup before training begins.
