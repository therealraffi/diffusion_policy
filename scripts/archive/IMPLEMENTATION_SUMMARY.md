# SLURM Baseline Scripts - Implementation Summary

## What Was Created

I've refactored your baseline training infrastructure to support SLURM HPC clusters. Here's what was implemented:

### 📁 New Scripts (in `/scripts/`)

```
scripts/
├── launch_all_baselines.sh     ⭐ MAIN LAUNCHER (new)
├── train_pusht.sh              ⭐ PushT task (new, SLURM-enabled)
├── train_can.sh                ⭐ Can task (new, SLURM-enabled)
├── train_transport.sh          ⭐ Transport task (new, SLURM-enabled)
├── train_and_eval.sh           ⭐ Train + evaluate (new, SLURM-enabled)
├── run_all_baselines.sh        (original - still works locally)
├── SLURM_README.md             📖 Full documentation
├── QUICK_START.md              📖 Quick reference
└── IMPLEMENTATION_SUMMARY.md   📖 This file
```

### 🎯 Design Decisions

| Aspect | Implementation |
|--------|-----------------|
| **Modularity** | 5 separate scripts + 1 launcher = mix and match |
| **Sequential vs Parallel** | Launcher supports both via `--parallel` flag |
| **GPU Allocation** | Individual GPUs per task (cuda:0) |
| **Resource Management** | 8 CPUs, 32 GB RAM per task, adaptive time limits |
| **Logging** | Timestamped SLURM output dirs + stdout/stderr separation |
| **Flexibility** | Can skip tasks, customize epochs, chain train+eval |
| **Documentation** | 3 docs (quick start, full guide, this summary) |

---

## How to Use

### 🚀 Immediate Start

```bash
cd /bigtemp/rhm4nj/safe_diffusion/diffusion_policy
sbatch scripts/launch_all_baselines.sh
```

### 📊 Monitor

```bash
squeue -u $USER
tail -f slurm_outputs/*.out
```

### ✅ Results

```
data/outputs/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_lowdim_<task>/
├── checkpoints/latest.ckpt
├── config.yaml
└── logs/
```

---

## All Usage Patterns

### Pattern 1: All Three Sequential
```bash
sbatch scripts/launch_all_baselines.sh
# Runs: PushT (1h) → Can (2h) → Transport (2-3h) = ~5-6h total
```

### Pattern 2: All Three Parallel
```bash
sbatch scripts/launch_all_baselines.sh --parallel
# Runs: PushT + Can + Transport simultaneously = ~3h total
```

### Pattern 3: Individual Tasks
```bash
sbatch scripts/train_pusht.sh
sbatch scripts/train_can.sh
sbatch scripts/train_transport.sh
```

### Pattern 4: Train + Evaluate
```bash
sbatch scripts/train_and_eval.sh pusht_lowdim 100
sbatch scripts/train_and_eval.sh can_lowdim 100
sbatch scripts/train_and_eval.sh transport_lowdim 80
# Each task trained and evaluated immediately
```

### Pattern 5: Selective Tasks
```bash
sbatch scripts/launch_all_baselines.sh --skip-can
sbatch scripts/launch_all_baselines.sh --pusht-only
sbatch scripts/launch_all_baselines.sh --transport-only
```

---

## Script Comparison

### `launch_all_baselines.sh`
**Purpose:** Master launcher
- Submits up to 3 training jobs
- Sequential (default) or parallel
- Optional task skipping
- Centralizes job submission

```bash
sbatch scripts/launch_all_baselines.sh [--parallel] [--skip-*]
```

### `train_pusht.sh` / `train_can.sh` / `train_transport.sh`
**Purpose:** Individual task training
- SLURM headers with resource specs
- Task-specific dataset validation
- Standalone submission
- Self-contained training logic

```bash
sbatch scripts/train_<task>.sh
```

### `train_and_eval.sh`
**Purpose:** Combined pipeline
- Train a task
- Find latest checkpoint
- Run evaluation immediately
- Generate eval_log.json

```bash
sbatch scripts/train_and_eval.sh <task> <epochs>
```

### `run_all_baselines.sh` (Original)
**Purpose:** Local/interactive training
- No SLURM headers
- Sequential execution
- Multi-GPU support (4 GPUs)
- Still fully functional

```bash
bash scripts/run_all_baselines.sh
```

---

## Key Features Implemented

### ✅ SLURM Integration
- Full `#SBATCH` directives
- GPU type specification (RTX 4000 Ada)
- CPU and memory allocation
- Time limits per task
- Output/error routing to timestamped files

### ✅ Flexibility
- Mix/match scripts
- Skip individual tasks
- Custom epoch counts
- Parallel or sequential
- Train-only or train+eval

### ✅ Robustness
- Dataset verification before training
- Checkpoint validation in eval script
- Exit code checking
- Informative error messages
- Detailed logging

### ✅ Documentation
- Quick start guide (2 minutes)
- Full technical guide (15 minutes)
- Inline script comments
- Troubleshooting section
- Performance expectations

---

## Resource Allocation Breakdown

### Per-Task Allocation
```
GPU:       1 × RTX 4000 Ada
CPU:       8 cores
Memory:    32 GB
Partition: gpu
```

### Time Limits (adaptive)
```
PushT:     4 hours   (100 epochs, 30 MB data)
Can:       6 hours   (100 epochs, 45 MB data)
Transport: 8 hours   (80 epochs, 309 MB data)
```

### Estimated Runtimes
```
PushT:     ~50 min training + 5 min eval = 55 min
Can:       ~100 min training + 10 min eval = 110 min
Transport: ~150 min training + 15 min eval = 165 min

Sequential total: ~5-6 hours
Parallel total:   ~3 hours (limited by longest task)
```

---

## File Structure After Submission

```
diffusion_policy/
├── data/
│   ├── outputs/
│   │   └── YYYY.MM.DD/
│   │       ├── HH.MM.SS_train_diffusion_unet_lowdim_pusht/
│   │       ├── HH.MM.SS_train_diffusion_unet_lowdim_can/
│   │       └── HH.MM.SS_train_diffusion_unet_lowdim_transport/
│   └── eval_results/  (if using train_and_eval.sh)
│       ├── eval_pusht_YYYYMMDD_HHMMSS/
│       ├── eval_can_YYYYMMDD_HHMMSS/
│       └── eval_transport_YYYYMMDD_HHMMSS/
├── scripts/
│   ├── launch_all_baselines.sh      (new)
│   ├── train_pusht.sh               (new)
│   ├── train_can.sh                 (new)
│   ├── train_transport.sh           (new)
│   ├── train_and_eval.sh            (new)
│   ├── run_all_baselines.sh         (original)
│   ├── SLURM_README.md              (new)
│   ├── QUICK_START.md               (new)
│   └── IMPLEMENTATION_SUMMARY.md    (new - this file)
└── slurm_outputs/
    ├── difpol_pusht-<JOB_ID>.out
    ├── difpol_pusht-<JOB_ID>.err
    ├── difpol_can-<JOB_ID>.out
    ├── difpol_can-<JOB_ID>.err
    ├── difpol_transport-<JOB_ID>.out
    ├── difpol_transport-<JOB_ID>.err
    └── ...
```

---

## Customization Examples

### Extend training time
Edit the script header:
```bash
#SBATCH --time=08:00:00  # 8 hours instead of 4
```

### Use different GPU type
```bash
#SBATCH --gpus=a100:1    # Request A100 instead of RTX 4000
```

### More CPU cores
```bash
#SBATCH --cpus-per-task=16  # 16 instead of 8
```

### More memory
```bash
#SBATCH --mem=64G         # 64 GB instead of 32 GB
```

### Custom job name for tracking
```bash
#SBATCH --job-name=baseline_v2   # Custom identifier
```

---

## Monitoring & Management

### Submit jobs
```bash
sbatch scripts/launch_all_baselines.sh
```

### Check status
```bash
squeue -u $USER
squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

### View logs
```bash
tail -f slurm_outputs/*.out              # Real-time
cat slurm_outputs/difpol_pusht-*.out     # Full log
```

### Cancel job
```bash
scancel <JOB_ID>
scancel -u $USER                         # Cancel all
```

### Job dependencies (advanced)
```bash
JID1=$(sbatch --parsable scripts/train_pusht.sh)
JID2=$(sbatch --dependency=afterok:$JID1 scripts/train_can.sh)
JID3=$(sbatch --dependency=afterok:$JID2 scripts/train_transport.sh)
```

---

## Comparison: Old vs New

### Local Training (Original)
```bash
bash scripts/run_all_baselines.sh
```
- ✅ Works locally
- ✅ Easy debugging
- ✅ 4 GPU parallelism
- ❌ Blocks terminal
- ❌ No HPC integration
- ❌ Manual monitoring

### HPC Cluster Training (New)
```bash
sbatch scripts/launch_all_baselines.sh
```
- ✅ HPC-native
- ✅ Queue management
- ✅ Resource reservation
- ✅ Frees terminal
- ✅ Logging to files
- ✅ Job tracking
- ❌ Requires SLURM cluster

---

## Next Steps

### 1. Submit baseline jobs
```bash
cd /bigtemp/rhm4nj/safe_diffusion/diffusion_policy
sbatch scripts/launch_all_baselines.sh --parallel
```

### 2. Monitor training
```bash
# Check every 2 minutes
watch -n 2 'squeue -u $USER'

# Or tail logs
tail -f slurm_outputs/*.out
```

### 3. Retrieve results
```bash
# After all jobs complete
ls data/outputs/YYYY.MM.DD/*/checkpoints/latest.ckpt
python multirun_metrics.py --result_dir data/
```

### 4. Further training
- Extend epochs: `train_and_eval.sh <task> <more_epochs>`
- Fine-tune hyperparams: Edit config files
- Chain jobs: Use SLURM dependencies

---

## Summary

| Aspect | Status |
|--------|--------|
| **SLURM scripts** | ✅ 5 scripts created |
| **Documentation** | ✅ Full + quick start |
| **Testing** | ✅ Syntax verified |
| **Execution** | Ready to `sbatch` |
| **Backward compatibility** | ✅ Original script still works |

**To start:**
```bash
sbatch /bigtemp/rhm4nj/safe_diffusion/diffusion_policy/scripts/launch_all_baselines.sh
```

For detailed info, see `SLURM_README.md` or `QUICK_START.md`.

