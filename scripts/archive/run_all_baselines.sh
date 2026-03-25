#!/bin/bash
# Safe Diffusion - Baseline Training and Evaluation Script
# Runs all three baseline models sequentially and evaluates them

set -e  # Exit on error

ENV="/bigtemp/rhm4nj/envs/safediff"
WORKSPACE="/bigtemp/rhm4nj/safe_diffusion/diffusion_policy"

export MUJOCO_GL=egl
export PYTHONPATH="$WORKSPACE:$PYTHONPATH"

cd "$WORKSPACE"

echo "======================================"
echo "Safe Diffusion - Baseline Setup"
echo "======================================"
echo ""

# Function to run training
run_training() {
    local TASK=$1
    local EXP_NAME=$2
    local NUM_EPOCHS=$3
    local GPU=$4

    echo "   Starting $TASK training..."
    echo "   Epochs: $NUM_EPOCHS"
    echo "   GPU: cuda:$GPU"
    echo ""

    conda run -p "$ENV" python -c "import _setup_env" && \
    conda run -p "$ENV" python _train_wrapper.py \
        --config-name=train_diffusion_unet_lowdim_workspace \
        task="$TASK" \
        exp_name="$EXP_NAME" \
        training.num_epochs="$NUM_EPOCHS" \
        training.rollout_every=10 \
        training.checkpoint_every=10 \
        training.device="cuda:$GPU" \
        logging.mode=offline

    echo "✅ $TASK training complete!"
    echo ""
}

# Function to run evaluation
run_eval() {
    local TASK=$1
    local OUTPUT_DIR=$2
    local GPU=$3

    echo "📊 Evaluating $TASK..."

    # Find latest checkpoint
    CKPT=$(find data/outputs -path "*${TASK}*" -name "latest.ckpt" | sort | tail -1)

    if [ -z "$CKPT" ]; then
        echo "❌ No checkpoint found for $TASK"
        return 1
    fi

    echo "   Using checkpoint: $CKPT"

    conda run -p "$ENV" python -c "import _setup_env" && \
    conda run -p "$ENV" python eval.py \
        --checkpoint "$CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --device "cuda:$GPU"

    echo "✅ $TASK evaluation complete!"
    echo ""
}

# Check if datasets exist
echo "Checking datasets..."
if [ ! -d "data/pusht/pusht_cchi_v7_replay.zarr" ]; then
    echo "❌ PushT dataset not found!"
    exit 1
fi
if [ ! -f "data/robomimic/datasets/can/ph/low_dim.hdf5" ]; then
    echo "❌ Can dataset not found!"
    exit 1
fi
if [ ! -f "data/robomimic/datasets/transport/ph/low_dim.hdf5" ]; then
    echo "❌ Transport dataset not found!"
    exit 1
fi
echo "✅ All datasets present"
echo ""

# Parse arguments
SKIP_PUSHT=false
SKIP_CAN=false
SKIP_TRANSPORT=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --skip-eval)
            SKIP_EVAL=true
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
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-pusht] [--skip-can] [--skip-transport] [--skip-eval]"
            echo "       [--pusht-only] [--can-only] [--transport-only]"
            exit 1
            ;;
    esac
done

# Training phase
echo "╔═════════════════════════════════════════╗"
echo "║ TRAINING PHASE                          ║"
echo "╚═════════════════════════════════════════╝"
echo ""

if [ "$SKIP_PUSHT" = false ]; then
    run_training "pusht_lowdim" "pusht_baseline_final" 100 0
else
    echo "⏭️ Skipping PushT training"
fi

if [ "$SKIP_CAN" = false ]; then
    run_training "can_lowdim" "can_baseline_final" 100 1
else
    echo "⏭️ Skipping Can training"
fi

if [ "$SKIP_TRANSPORT" = false ]; then
    run_training "transport_lowdim" "transport_baseline_final" 80 2
else
    echo "⏭️ Skipping Transport training"
fi

if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "╔═════════════════════════════════════════╗"
    echo "║ EVALUATION PHASE                        ║"
    echo "╚═════════════════════════════════════════╝"
    echo ""

    if [ "$SKIP_PUSHT" = false ]; then
        run_eval "pusht_baseline" "data/eval_pusht_final" 0
    fi

    if [ "$SKIP_CAN" = false ]; then
        run_eval "can_baseline" "data/eval_can_final" 1
    fi

    if [ "$SKIP_TRANSPORT" = false ]; then
        run_eval "transport_baseline" "data/eval_transport_final" 2
    fi

    echo ""
    echo "╔═════════════════════════════════════════╗"
    echo "║ METRICS AGGREGATION                     ║"
    echo "╚═════════════════════════════════════════╝"
    echo ""

    conda run -p "$ENV" python -c "import _setup_env" && \
    conda run -p "$ENV" python multirun_metrics.py \
        --result_dir data/ \
        --output_file results_baseline_summary.json

    echo "✅ Metrics aggregated!"
    echo "   Summary: results_baseline_summary.json"
    echo ""
fi

echo "🎉 All done!"
echo ""
echo "Generated outputs:"
echo "  - Training: data/outputs/"
echo "  - Evaluation: data/eval_*/"
if [ "$SKIP_EVAL" = false ]; then
    echo "  - Metrics: results_baseline_summary.json"
fi
