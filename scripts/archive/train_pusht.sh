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
# Safe Diffusion - PushT Baseline Training (SLURM)
# Task: 2D Navigation in simulation (state-based observations)
# Dataset: ~30MB zarr format
# Expected runtime: ~45-60 minutes for 100 epochs
# ============================================================================

module purge
module load miniforge
module load gcc
module load cuda

nvidia-smi

source ~/.bashrc
export MPLBACKEND=agg
export HYDRA_FULL_ERROR=1

# Setup
ENV="/bigtemp/rhm4nj/envs/safediff"
WORKSPACE="/bigtemp/rhm4nj/safe_diffusion/diffusion_policy"

OUTPUT_DIR="data/outputs/slurm_runs/pusht_job${SLURM_JOB_ID}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$WORKSPACE:$PYTHONPATH"

cd "$WORKSPACE"

# Create output directory for this job
mkdir -p slurm_outputs

# Print job info
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Task: PushT (2D Navigation, State-based)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CPUs: $SLURM_CPUS_PER_TASK | Memory: $SLURM_MEM_PER_NODE MB"
echo "Working dir: $WORKSPACE"
echo ""

# Verify datasets
if [ ! -d "data/pusht/pusht_cchi_v7_replay.zarr" ]; then
    echo "❌ PushT dataset not found at data/pusht/pusht_cchi_v7_replay.zarr"
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV"
echo 'using Python:'
which python

# Set explicit output directory
mkdir -p "$OUTPUT_DIR"

echo 'Starting Train...'
python train_with_setup.py \
    --config-name=train_diffusion_unet_lowdim_workspace \
    task=pusht_lowdim_no_runner \
    exp_name="pusht_job${SLURM_JOB_ID}" \
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
