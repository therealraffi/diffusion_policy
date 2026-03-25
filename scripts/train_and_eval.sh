#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=difpol_pusht
#SBATCH --output=/bigtemp/rhm4nj/safe_diffusion/diffusion_policy/slurm_outputs/%x-%j.out
#SBATCH --error=/bigtemp/rhm4nj/safe_diffusion/diffusion_policy/slurm_outputs/%x-%j.err

# ============================================================================
# Safe Diffusion - Training + Evaluation (SLURM)
# Trains a single task and immediately evaluates the result
#
# Usage:
#   sbatch train_and_eval.sh pusht_lowdim 100
#   sbatch train_and_eval.sh can_lowdim 100
#   sbatch train_and_eval.sh transport_lowdim 80
# ============================================================================

set -e

module purge
module load miniforge
module load gcc
module load cuda

nvidia-smi

source ~/.bashrc
export MPLBACKEND=agg
export HYDRA_FULL_ERROR=1

ENV="/bigtemp/rhm4nj/envs/safediff"
WORKSPACE="/bigtemp/rhm4nj/safe_diffusion/diffusion_policy"

export MUJOCO_GL=egl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$WORKSPACE:$PYTHONPATH"

# Parse arguments (default: PushT with 100 epochs)
TASK=${1:-pusht_lowdim}
EPOCHS=${2:-100}

# Extract short task name (e.g., "pusht" from "pusht_lowdim")
TASK_SHORT=$(echo "$TASK" | cut -d'_' -f1)

cd "$WORKSPACE"

# Activate conda environment
# Note: Using full path to python instead of 'conda activate' for SLURM compatibility
# conda activate doesn't work reliably in SLURM job context
PYTHON="$ENV/bin/python"
echo "using Python:"
echo "$PYTHON"
$PYTHON --version
which pip

# Create output directories
mkdir -p slurm_outputs
mkdir -p data/eval_results

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Safe Diffusion - Training + Evaluation (SLURM)            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Task: $TASK"
echo "Epochs: $EPOCHS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CPUs: $SLURM_CPUS_PER_TASK | Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# ============================================================================
# TRAINING PHASE
# ============================================================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║ TRAINING PHASE                                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Starting training for $TASK..."
echo ""

$PYTHON train_with_setup.py \
    --config-name=train_diffusion_unet_lowdim_workspace \
    task="$TASK" \
    exp_name="${TASK_SHORT}_train_eval" \
    training.num_epochs="$EPOCHS" \
    training.rollout_every=10 \
    training.checkpoint_every=10 \
    training.device=cuda:0 \
    logging.mode=offline

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo ""
echo "✅ Training complete!"
echo ""

# ============================================================================
# EVALUATION PHASE
# ============================================================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║ EVALUATION PHASE                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Starting evaluation..."
echo ""

# Find latest checkpoint for this task
CKPT=$(find data/outputs -path "*${TASK_SHORT}*" -name "latest.ckpt" | sort | tail -1)

if [ -z "$CKPT" ]; then
    echo "❌ No checkpoint found for task '$TASK_SHORT'"
    echo "   Searched in data/outputs/ for '*${TASK_SHORT}*' + 'latest.ckpt'"
    exit 1
fi

echo "Found checkpoint: $CKPT"
echo ""

# Create evaluation output directory
EVAL_OUTPUT="data/eval_results/eval_${TASK_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_OUTPUT"

echo "Running evaluation..."
echo "Output directory: $EVAL_OUTPUT"
echo ""

$PYTHON eval.py --checkpoint "$CKPT" --output_dir "$EVAL_OUTPUT" --device cuda:0

EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "❌ Evaluation failed with exit code $EVAL_EXIT"
    exit $EVAL_EXIT
fi

echo ""
echo "✅ Evaluation complete!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║ SUMMARY                                                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Task: $TASK"
echo "Epochs: $EPOCHS"
echo "Training checkpoint: $CKPT"
echo "Evaluation results: $EVAL_OUTPUT/eval_log.json"
echo ""

if [ -f "$EVAL_OUTPUT/eval_log.json" ]; then
    echo "📋 Evaluation metrics:"
    cat "$EVAL_OUTPUT/eval_log.json" | $PYTHON -m json.tool 2>/dev/null || cat "$EVAL_OUTPUT/eval_log.json"
    echo ""
fi

echo "🎉 Train + Eval pipeline complete!"
echo ""
