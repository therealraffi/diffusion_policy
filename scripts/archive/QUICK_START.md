# SLURM Baseline Scripts - Quick Start Guide

## 5-Second Start

```bash
cd /bigtemp/rhm4nj/safe_diffusion/diffusion_policy
sbatch scripts/launch_all_baselines.sh
```

Monitor:
```bash
squeue -u $USER
tail -f slurm_outputs/*.out
```

## Files Created

| File | Purpose | Type |
|------|---------|------|
| `launch_all_baselines.sh` | Submit all 3 jobs | Launcher |
| `train_pusht.sh` | Train PushT task | Individual task |
| `train_can.sh` | Train Can task | Individual task |
| `train_transport.sh` | Train Transport task | Individual task |
| `train_and_eval.sh` | Train + evaluate single task | Utility |
| `SLURM_README.md` | Full documentation | Documentation |

## Common Usage Patterns

### 1️⃣ Submit All Three Tasks (Sequentially)
```bash
sbatch scripts/launch_all_baselines.sh
# Takes ~5 hours total, one after another
```

### 2️⃣ Submit All Three Tasks (Parallel)
```bash
sbatch scripts/launch_all_baselines.sh --parallel
# Takes ~3 hours total, all running together
```

### 3️⃣ Train Individual Task
```bash
sbatch scripts/train_pusht.sh      # ~1 hour
sbatch scripts/train_can.sh        # ~2 hours
sbatch scripts/train_transport.sh  # ~2-3 hours
```

### 4️⃣ Train + Immediate Evaluation
```bash
sbatch scripts/train_and_eval.sh pusht_lowdim 100
sbatch scripts/train_and_eval.sh can_lowdim 100
sbatch scripts/train_and_eval.sh transport_lowdim 80
# Each includes training + evaluation
```

### 5️⃣ Skip Some Tasks
```bash
sbatch scripts/launch_all_baselines.sh --skip-can
sbatch scripts/launch_all_baselines.sh --pusht-only
sbatch scripts/launch_all_baselines.sh --skip-pusht --skip-transport
```

## Monitor Jobs

```bash
# Check status
squeue -u $USER

# Detailed status
squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Watch logs (live)
tail -f slurm_outputs/*.out

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
squeue -u $USER -h | awk '{print $1}' | xargs scancel
```

## Where to Find Results

```bash
# Training outputs
ls -la data/outputs/

# Evaluation results (if using train_and_eval.sh)
ls -la data/eval_results/

# Job logs
ls -la slurm_outputs/
```

## Key Features

✅ **SLURM-ready** - Works on HPC clusters
✅ **GPU allocation** - RTX 4000 Ada GPUs specified
✅ **Resource management** - 8 CPUs, 32 GB RAM, proper time limits
✅ **Logging** - All outputs to timestamped files
✅ **Flexible** - Run one, some, or all tasks
✅ **Parallel or sequential** - Choose your workflow
✅ **Train+Eval** - Combined pipeline for quick iteration

## Resources per Task

| Task | GPU | CPU | RAM | Time | Data |
|------|-----|-----|-----|------|------|
| PushT | 1 | 8 | 32G | 4h | 30 MB |
| Can | 1 | 8 | 32G | 6h | 45 MB |
| Transport | 1 | 8 | 32G | 8h | 309 MB |

## Troubleshooting

**"Permission denied"** → Make executable:
```bash
chmod +x scripts/*.sh
```

**"Dataset not found"** → Verify they exist:
```bash
ls data/pusht/pusht_cchi_v7_replay.zarr/
ls data/robomimic/datasets/can/ph/low_dim.hdf5
ls data/robomimic/datasets/transport/ph/low_dim.hdf5
```

**"Module not found (gym, etc.)"** → Install dependencies:
```bash
conda run -p /bigtemp/rhm4nj/envs/safediff pip install gym robomimic
```

**Job timed out** → Increase time in script header:
```bash
#SBATCH --time=10:00:00  # Increase from 4h to 10h
```

## Full Documentation

See `SLURM_README.md` for detailed information about:
- All SLURM directives explained
- Output structure and checkpoints
- Performance expectations
- Advanced customization
- Integration notes

---

**Start now:**
```bash
cd /bigtemp/rhm4nj/safe_diffusion/diffusion_policy
sbatch scripts/launch_all_baselines.sh
```

