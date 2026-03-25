# Baseline Training Status

## ✅ Completed and Working

### Core Fixes Applied
1. **Workspace Modified**: `train_diffusion_unet_lowdim_workspace.py` now conditionally instantiates `env_runner`
   - Lines 108-113: Check if `env_runner` exists in config before instantiation
   - Line 217: Skip rollout execution if `env_runner` is None
   - Lines 284-296: Provide fallback metrics for checkpoint selection when rollouts are disabled

2. **Setup Script Created**: `train_with_setup.py`
   - Imports `_setup_env` before training to ensure gymnasium/gym compatibility in same Python process
   - Solves issue where separate conda run commands were in different shells

3. **Compatibility Module**: `_setup_env.py`
   - Aliases gymnasium to gym for backward compatibility
   - Allows old code to work with new gymnasium library

### PushT Task ✅ FULLY WORKING
- **Config**: `diffusion_policy/config/task/pusht_lowdim_no_runner.yaml`
- **Dataset**: PushT 2D navigation (zarr format)
- **Training Command**:
  ```bash
  conda run -p /bigtemp/rhm4nj/envs/safediff python train_with_setup.py \
    --config-name=train_diffusion_unet_lowdim_workspace \
    task=pusht_lowdim_no_runner \
    exp_name=pusht_baseline \
    training.num_epochs=100 \
    training.checkpoint_every=10 \
    training.device=cuda:0 \
    logging.mode=offline
  ```

**Verified:** Training runs without errors, creates checkpoints, logs metrics

### SLURM Scripts Created
All scripts use RTX 4000 Ada GPU, 8 CPUs, 32GB memory:

- **train_pusht.sh**: 4-hour time limit, 100 epochs
- **train_can.sh**: 6-hour time limit, 100 epochs
- **train_transport.sh**: 8-hour time limit, 80 epochs
- **launch_all_baselines.sh**: Master launcher with --parallel flag support

**Usage**: `sbatch scripts/train_pusht.sh` (outputs go to `slurm_outputs/`)

---

## ⚠️ Known Issues

### Can and Transport Tasks
- **Issue**: Require `pytorch3d` library for rotation transformation in RobomimicReplayLowdimDataset
- **Status**: Dependency installation failing due to conda issues
- **Workaround Options**:
  1. Install pytorch3d manually: `pip install pytorch3d` or via conda-forge
  2. Use simplified dataset (requires code changes to robomimic dataset)
  3. Skip rollout evaluation as done in PushT

**Configs Created**: Both `can_lowdim_no_runner.yaml` and `transport_lowdim_no_runner.yaml` ready once pytorch3d is available

---

## 📋 Summary

| Task | Status | Notes |
|------|--------|-------|
| PushT | ✅ Ready | Tested and verified working |
| Can | ⏳ Blocked | Needs pytorch3d |
| Transport | ⏳ Blocked | Needs pytorch3d |
| Workspace Fix | ✅ Done | Handles missing env_runner |
| SLURM Scripts | ✅ Done | All three baseline scripts created |
| Setup Module | ✅ Done | Gymnasium/gym compatibility solved |

---

## 🚀 Next Steps

1. **For PushT**: Ready to submit SLURM job
   ```bash
   sbatch scripts/train_pusht.sh
   ```

2. **For Can/Transport**: Install pytorch3d dependency
   ```bash
   conda install -c fvcore -c iopath -c conda-forge pytorch3d
   ```
   Then test with:
   ```bash
   sbatch scripts/train_can.sh
   ```

3. **Monitor Training**:
   ```bash
   tail -f slurm_outputs/difpol_pusht-<JOBID>.out
   ```

4. **Find Results**:
   - Checkpoints: `data/outputs/YYYY.MM.DD/HH.MM.SS_*/checkpoints/`
   - Logs: `slurm_outputs/<job-name>-<job-id>.out`
   - Wandb offline: `data/outputs/*/wandb/offline-run-*/`
