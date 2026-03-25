# Safe Diffusion - SLURM Baseline Training Scripts

This directory contains SLURM-compatible scripts for training baseline diffusion models on three robotic tasks on HPC clusters.

## Overview

| Script | Purpose | Tasks | GPU Time |
|--------|---------|-------|----------|
| `launch_all_baselines.sh` | Submit all 3 training jobs (one launcher) | PushT, Can, Transport | 4-8 hrs |
| `train_pusht.sh` | Individual task: PushT training | PushT only | ~1 hr |
| `train_can.sh` | Individual task: Can training | Can only | ~2 hrs |
| `train_transport.sh` | Individual task: Transport training | Transport only | ~2-3 hrs |
| `train_and_eval.sh` | Train + evaluate a single task | Any task | ~3-4 hrs |

## Quick Start

### Option 1: Submit all three tasks (Recommended)

```bash
# Sequential submission (waits for each to complete)
cd /bigtemp/rhm4nj/safe_diffusion/diffusion_policy
sbatch scripts/launch_all_baselines.sh

# OR: Parallel submission (all run at same time)
sbatch scripts/launch_all_baselines.sh --parallel
```

### Option 2: Submit individual tasks

```bash
sbatch scripts/train_pusht.sh
sbatch scripts/train_can.sh
sbatch scripts/train_transport.sh
```

### Option 3: Train + Evaluate (immediate feedback)

```bash
# Train PushT with 100 epochs and evaluate
sbatch scripts/train_and_eval.sh pusht_lowdim 100

# Train Can with 100 epochs and evaluate
sbatch scripts/train_and_eval.sh can_lowdim 100

# Train Transport with 80 epochs and evaluate
sbatch scripts/train_and_eval.sh transport_lowdim 80
```

## Monitoring Jobs

```bash
# View your queued/running jobs
squeue -u $USER

# Detailed view with more columns
squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Watch logs in real-time
tail -f slurm_outputs/*.out

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
squeue -u $USER -h | awk '{print $1}' | xargs scancel
```

## Script Details

### `launch_all_baselines.sh`

**Purpose:** Master launcher that submits all three training jobs

**Usage:**
```bash
./scripts/launch_all_baselines.sh              # Sequential (default)
./scripts/launch_all_baselines.sh --parallel   # Parallel
./scripts/launch_all_baselines.sh --skip-can   # Skip Can task
./scripts/launch_all_baselines.sh --pusht-only # Train only PushT
```

**Resource Allocation:**
- GPU: 1× RTX 4000 Ada per task
- CPUs: 8 per task
- Memory: 32 GB per task
- Time limit: 4 hrs (PushT), 6 hrs (Can), 8 hrs (Transport)

**Output:**
- Logs: `slurm_outputs/job_name-JOB_ID.{out,err}`
- Models: `data/outputs/YYYY.MM.DD/HH.MM.SS_*_task_name/`

### `train_pusht.sh`

**Task:** 2D navigation in simulation (state-based)
- **Dataset:** 30 MB zarr format
- **Expected time:** 45-60 minutes (100 epochs)
- **Epochs:** 100
- **GPU:** cuda:0

```bash
sbatch scripts/train_pusht.sh
```

### `train_can.sh`

**Task:** Single-arm manipulation - pick and place a can
- **Dataset:** 45 MB HDF5 (Robomimic)
- **Expected time:** 90-120 minutes (100 epochs)
- **Epochs:** 100
- **GPU:** cuda:0

```bash
sbatch scripts/train_can.sh
```

### `train_transport.sh`

**Task:** Dual-arm coordination - pick and transport
- **Dataset:** 309 MB HDF5 (Robomimic)
- **Expected time:** 120-180 minutes (80 epochs)
- **Epochs:** 80
- **GPU:** cuda:0

```bash
sbatch scripts/train_transport.sh
```

### `train_and_eval.sh`

**Purpose:** Train a model and immediately run evaluation

**Usage:**
```bash
# Train with custom epochs
sbatch scripts/train_and_eval.sh <TASK> <EPOCHS>

# Examples:
sbatch scripts/train_and_eval.sh pusht_lowdim 100
sbatch scripts/train_and_eval.sh can_lowdim 100
sbatch scripts/train_and_eval.sh transport_lowdim 80
```

**Output:**
```
data/eval_results/eval_<task>_YYYYMMDD_HHMMSS/
├── eval_log.json           # Evaluation metrics
├── *.mp4                   # Video rollouts (if enabled)
└── renders/                # Rendered images (if enabled)
```

## SLURM Directives Explained

Each training script includes:

```bash
#SBATCH --partition=gpu          # GPU partition
#SBATCH --gres=gpu:1             # 1 GPU resource
#SBATCH --gpus=rtx_4000_ada:1    # Request specific GPU model
#SBATCH --cpus-per-task=8        # 8 CPU cores
#SBATCH --mem=32G                # 32 GB RAM
#SBATCH --time=HH:MM:SS          # Time limit
#SBATCH --job-name=difpol_*      # Job name for easy identification
#SBATCH --output=...             # stdout file
#SBATCH --error=...              # stderr file
```

### Customizing Resources

Edit the `#SBATCH` directives in any script:

```bash
# More GPU memory? Increase time:
#SBATCH --time=10:00:00          # 10 hours instead of 4

# Need more CPUs? Increase batch size:
#SBATCH --cpus-per-task=16       # 16 CPUs instead of 8

# Need more RAM:
#SBATCH --mem=64G                # 64 GB instead of 32 GB
```

## Output Structure

```
diffusion_policy/
├── data/
│   ├── outputs/
│   │   └── YYYY.MM.DD/
│   │       └── HH.MM.SS_train_diffusion_unet_lowdim_<task>/
│   │           ├── checkpoints/
│   │           │   ├── epoch_0010_checkpoint.ckpt
│   │           │   └── latest.ckpt
│   │           ├── logs/
│   │           └── config.yaml
│   └── eval_results/
│       └── eval_<task>_YYYYMMDD_HHMMSS/
│           ├── eval_log.json
│           └── videos/  (if enabled)
└── slurm_outputs/
    ├── difpol_pusht-<JOB_ID>.out
    ├── difpol_pusht-<JOB_ID>.err
    └── ...
```

## Practical Workflows

### Workflow 1: Full Baseline Suite (Recommended)

Train all three tasks sequentially, then evaluate:

```bash
# Submit all three
sbatch scripts/launch_all_baselines.sh

# Monitor
watch squeue -u $USER

# Once all complete, run evaluation separately
sbatch scripts/train_and_eval.sh pusht_lowdim 0    # 0 epochs = eval only
sbatch scripts/train_and_eval.sh can_lowdim 0
sbatch scripts/train_and_eval.sh transport_lowdim 0
```

### Workflow 2: Parallel Training

All three tasks train simultaneously on different GPUs:

```bash
sbatch scripts/launch_all_baselines.sh --parallel

# Monitor
squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Takes ~3 hours total (longest task: transport)
```

### Workflow 3: Single Task with Tuning

Train one task, evaluate, then tune hyperparameters:

```bash
# Train and evaluate
sbatch scripts/train_and_eval.sh pusht_lowdim 50

# Wait for results, check performance
tail -f slurm_outputs/difpol_train_eval-*.out

# If good, train longer
sbatch scripts/train_and_eval.sh pusht_lowdim 200
```

## Troubleshooting

### Job fails with "No module named 'gym'"

The conda environment is missing dependencies. Run:
```bash
cd /bigtemp/rhm4nj/safe_diffusion/diffusion_policy
conda run -p /bigtemp/rhm4nj/envs/safediff pip install gym robomimic
```

### Job fails with "Dataset not found"

Verify datasets are in place:
```bash
# PushT
ls -lh data/pusht/pusht_cchi_v7_replay.zarr/

# Can
ls -lh data/robomimic/datasets/can/ph/low_dim.hdf5

# Transport
ls -lh data/robomimic/datasets/transport/ph/low_dim.hdf5
```

### GPU out of memory

Increase memory and time limits in the script:
```bash
#SBATCH --mem=64G        # Double memory
#SBATCH --time=10:00:00  # Increase time
```

Or reduce batch size in config (edit `train.py` arguments).

### Job times out

The default time limits are:
- PushT: 4 hours
- Can: 6 hours
- Transport: 8 hours

For longer training, increase in script:
```bash
#SBATCH --time=12:00:00  # 12 hours
```

## Performance Expectations

Training on RTX 4000 Ada GPUs:

| Task | Dataset | Epochs | Time | Final Loss |
|------|---------|--------|------|-----------|
| PushT | 30 MB | 100 | ~50 min | ~0.005 |
| Can | 45 MB | 100 | ~100 min | ~0.008 |
| Transport | 309 MB | 80 | ~150 min | ~0.012 |

## Advanced Usage

### Custom training duration

Edit the epoch count in launcher or train_and_eval.sh:

```bash
# In launch_all_baselines.sh, change:
run_training "pusht_lowdim" "pusht_baseline_final" 200 0  # 200 epochs
```

### Custom GPU allocation

Request multiple GPUs for data parallelism:

```bash
# In any script, change:
#SBATCH --gres=gpu:2        # 2 GPUs
```

Then update training script:
```bash
# Use torch.nn.DataParallel or similar
```

### Generate videos during training

Edit the task config or add to train call:
```bash
training.save_video=true \
training.video_fps=30
```

## Integration with Existing Scripts

These SLURM scripts complement the existing `run_all_baselines.sh`:

- **`run_all_baselines.sh`**: Local/interactive training
- **`launch_all_baselines.sh`**: HPC cluster submission

Choose based on your compute environment.

## Support

For issues or questions:
1. Check logs: `cat slurm_outputs/*.out`
2. Verify dataset paths with `ls` commands above
3. Test locally first with `bash scripts/run_all_baselines.sh`
4. Check SLURM syntax: `man sbatch`

