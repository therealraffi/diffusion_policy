#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=difpol_can
#SBATCH --output=/bigtemp/rhm4nj/safe_diffusion/diffusion_policy/slurm_outputs/%x-%j.out
#SBATCH --error=/bigtemp/rhm4nj/safe_diffusion/diffusion_policy/slurm_outputs/%x-%j.err

# ============================================================================
# Safe Diffusion - Can (PickPlaceCan) Baseline Training (SLURM)
# Task: Single-arm manipulation - pick and place a can into a bin
# Dataset: ~45MB HDF5 format (Robomimic)
# Expected runtime: ~90-120 minutes for 100 epochs
# ============================================================================

set -e

module purge
module load cuda

nvidia-smi

source ~/.bashrc
export MPLBACKEND=agg

# Setup
ENV="/bigtemp/rhm4nj/envs/safediff"
WORKSPACE="/bigtemp/rhm4nj/safe_diffusion/diffusion_policy"

export MUJOCO_GL=egl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$WORKSPACE:$PYTHONPATH"

cd "$WORKSPACE"

# Create output directory for this job
mkdir -p slurm_outputs

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Safe Diffusion - Can (PickPlaceCan) Baseline Training     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Task: Can/PickPlaceCan (Single-arm Manipulation)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CPUs: $SLURM_CPUS_PER_TASK | Memory: $SLURM_MEM_PER_NODE MB"
echo "Working dir: $WORKSPACE"
echo ""

# Verify datasets
if [ ! -f "data/robomimic/datasets/can/ph/low_dim.hdf5" ]; then
    echo "❌ Can dataset not found at data/robomimic/datasets/can/ph/low_dim.hdf5"
    exit 1
fi
echo "✅ Dataset verified"
echo ""

source "$ENV/bin/activate"

# Set explicit output directory
OUTPUT_DIR="data/outputs/slurm_runs/can_job${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

python train_with_setup.py \
    --config-name=train_diffusion_unet_lowdim_workspace \
    task=can_lowdim_no_runner \
    exp_name="can_job${SLURM_JOB_ID}" \
    training.num_epochs=100 \
    training.checkpoint_every=10 \
    training.device=cuda:0 \
    logging.mode=offline \
    hydra.run.dir="$OUTPUT_DIR"

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# Run evaluation using same output directory
LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR/checkpoints"/epoch*.ckpt | head -1)
EVAL_OUTPUT="$OUTPUT_DIR/eval_results"

python eval.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --output_dir "$EVAL_OUTPUT" \
    --device cuda:0

EVAL_EXIT=$?
if [ $EVAL_EXIT -ne 0 ]; then
    echo "❌ Evaluation failed with exit code $EVAL_EXIT"
    exit $EVAL_EXIT
fi

echo "✅ Done: $OUTPUT_DIR"
