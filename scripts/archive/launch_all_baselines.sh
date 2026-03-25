#!/bin/bash
# Safe Diffusion - SLURM Job Launcher
# Submits training jobs for all three baselines to the cluster
#
# Usage:
#   ./launch_all_baselines.sh              # Sequential submission
#   ./launch_all_baselines.sh --parallel   # Parallel submission
#   ./launch_all_baselines.sh --skip-can   # Skip Can training
#

set -e

WORKSPACE="/bigtemp/rhm4nj/safe_diffusion/diffusion_policy"
SCRIPT_DIR="$WORKSPACE/scripts"

# Create timestamped output directories
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_BASE="$WORKSPACE/slurm_outputs/$TIMESTAMP"
mkdir -p "$OUTPUT_BASE"/{out,err}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Safe Diffusion - SLURM Baseline Training Launcher         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUTPUT_BASE"
echo ""

# Parse arguments
PARALLEL=false
SKIP_PUSHT=false
SKIP_CAN=false
SKIP_TRANSPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --skip-pusht)
            SKIP_PUSHT=true
            shift
            ;;
        --skip-can)
            SKIP_CAN=true
            shift
            ;;
        --skip-transport)
            SKIP_TRANSPORT=true
            shift
            ;;
        --pusht-only)
            SKIP_CAN=true
            SKIP_TRANSPORT=true
            shift
            ;;
        --can-only)
            SKIP_PUSHT=true
            SKIP_TRANSPORT=true
            shift
            ;;
        --transport-only)
            SKIP_PUSHT=true
            SKIP_CAN=true
            shift
            ;;
        *)
            echo "Usage: $0 [--parallel] [--skip-pusht] [--skip-can] [--skip-transport]"
            echo "          [--pusht-only] [--can-only] [--transport-only]"
            exit 1
            ;;
    esac
done

# Function to submit a job
submit_job() {
    local TASK=$1
    local SCRIPT=$2

    echo "📤 Submitting $TASK training..."

    if [ "$PARALLEL" = true ]; then
        # Non-blocking submit
        sbatch "$SCRIPT"
    else
        # Blocking submit (wait for completion)
        sbatch --wait "$SCRIPT"
    fi
}

# Verify datasets exist
echo "✓ Verifying datasets..."
if [ ! -d "$WORKSPACE/data/pusht/pusht_cchi_v7_replay.zarr" ] && [ "$SKIP_PUSHT" = false ]; then
    echo "❌ PushT dataset not found at $WORKSPACE/data/pusht/pusht_cchi_v7_replay.zarr"
    exit 1
fi
if [ ! -f "$WORKSPACE/data/robomimic/datasets/can/ph/low_dim.hdf5" ] && [ "$SKIP_CAN" = false ]; then
    echo "❌ Can dataset not found at $WORKSPACE/data/robomimic/datasets/can/ph/low_dim.hdf5"
    exit 1
fi
if [ ! -f "$WORKSPACE/data/robomimic/datasets/transport/ph/low_dim.hdf5" ] && [ "$SKIP_TRANSPORT" = false ]; then
    echo "❌ Transport dataset not found at $WORKSPACE/data/robomimic/datasets/transport/ph/low_dim.hdf5"
    exit 1
fi
echo "✅ All required datasets verified"
echo ""

# Submit jobs
if [ "$SKIP_PUSHT" = false ]; then
    submit_job "PushT" "$SCRIPT_DIR/train_pusht.sh"
else
    echo "⏭️  Skipping PushT training"
fi

if [ "$SKIP_CAN" = false ]; then
    submit_job "Can" "$SCRIPT_DIR/train_can.sh"
else
    echo "⏭️  Skipping Can training"
fi

if [ "$SKIP_TRANSPORT" = false ]; then
    submit_job "Transport" "$SCRIPT_DIR/train_transport.sh"
else
    echo "⏭️  Skipping Transport training"
fi

if [ "$PARALLEL" = true ]; then
    echo ""
    echo "⏳ Jobs submitted in parallel. Waiting for completion..."
    squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" | head -20
fi

echo ""
echo "✅ Job submission complete!"
echo "Monitor with: squeue -u \$USER"
echo "Check logs: $OUTPUT_BASE/{out,err}/"
echo ""
