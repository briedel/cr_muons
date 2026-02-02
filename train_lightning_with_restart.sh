#!/bin/bash
# Robust training launcher for PyTorch Lightning implementation
# Handles Pelican federation connection issues and automatic restarts
# Iterates through directory ranges (0000000-0000999, 0001000-0001999, etc.)

set -o pipefail

# --- Configuration ---
MAX_RETRIES=0           # 0 = infinite retries
INITIAL_BACKOFF=5       # seconds
MAX_BACKOFF=300         # maximum backoff (5 minutes)
BACKOFF_MULTIPLIER=2

# Range configuration for directory iteration
RANGE_START=0
RANGE_END=99000
RANGE_STEP=1000
CURRENT_RANGE_FILE=".current_range_lightning"
RANGES_TO_PROCESS=2     # Limit to next N ranges (set 0 for unlimited)

# Logging
LOG_DIR="./logs_tensorboard"
RUN_LOG_DIR="./logs_training_lightning"
ERROR_LOG="${RUN_LOG_DIR}/training_errors.log"

mkdir -p "$LOG_DIR"
mkdir -p "$RUN_LOG_DIR"

# Load or initialize range
if [ -f "$CURRENT_RANGE_FILE" ]; then
    CURRENT_RANGE=$(cat "$CURRENT_RANGE_FILE")
else
    CURRENT_RANGE=$RANGE_START
fi

if [ "$RANGES_TO_PROCESS" -gt 0 ]; then
    TARGET_END=$(( CURRENT_RANGE + RANGE_STEP * (RANGES_TO_PROCESS - 1) ))
    if [ "$TARGET_END" -lt "$RANGE_END" ]; then RANGE_END=$TARGET_END; fi
fi

echo "=== Lightning Training Launcher with Auto-Restart ===" | tee -a "$ERROR_LOG"
echo "Start time:    $(date)" | tee -a "$ERROR_LOG"
echo "Range:         ${RANGE_START} to ${RANGE_END} (step: ${RANGE_STEP})" | tee -a "$ERROR_LOG"
echo "Target script: src/train.py" | tee -a "$ERROR_LOG"
echo "" | tee -a "$ERROR_LOG"

# --- Iterate through ranges ---
while [ "$CURRENT_RANGE" -le "$RANGE_END" ]; do
    RANGE_LOW=$(printf "%07d" $CURRENT_RANGE)
    RANGE_HIGH=$(printf "%07d" $((CURRENT_RANGE + RANGE_STEP - 1)))
    RANGE_DIR="${RANGE_LOW}-${RANGE_HIGH}"
    
    echo "[$(date)] 📁 Processing range: $RANGE_DIR" | tee -a "$ERROR_LOG"
    
    # Build training command for Lightning
    # Note: Using underscores for arguments as defined in src/train.py
    TRAIN_CMD=(
        # stdbuf -oL -eL \ 
        python3 -u src/train.py \
        --model_type gan \
        --data_dir "pelican://osg-htc.org/icecube/wipac/data/sim/IceCube/2025/testing/${RANGE_DIR}/*.parquet" \
        --file_format parquet \
        --batch_size 128000 \
        --num_workers 4 \
        --prefetch_factor 4 \
        --pin_memory \
        --multi_file_shuffle 10 \
        --prefetch_batches 100 \
        --prefetch_ahead 20 \
        --prefetch_dir "./testdata/" \
        --auto_token \
        --resume_last \
        --max_muons_per_event 50000 \
        --preflight_muon_threshold 300000 \
        --max_muons_per_batch 0 \
        --outliers_dir "./outliers_parquet/" \
        --tb_logdir "$LOG_DIR" \
        --tb_log_interval 20 \
        --tb_hist_interval 100 \
        --lr 0.5e-4 \
        --grad_clip_norm 0.0 \
        --lambda_gp 10.0 \
        --gp_every 2 \
        --gp_max_pairs 4096 \
        --accelerator mps \
        --devices 1
    )
    
    attempt=0
    backoff=$INITIAL_BACKOFF
    
    # --- Retry loop for current range ---
    while true; do
        attempt=$((attempt + 1))
        
        if [ "$MAX_RETRIES" -gt 0 ] && [ "$attempt" -gt "$MAX_RETRIES" ]; then
            echo "[$(date)] ❌ Max retries reached for $RANGE_DIR. Skipping." | tee -a "$ERROR_LOG"
            break
        fi
        
        echo "[$(date)] 🚀 Range $RANGE_DIR - Attempt $attempt" | tee -a "$ERROR_LOG"
        
        # Unique log file for this attempt
        LOG_PREFIX="${RUN_LOG_DIR}/lightning_${RANGE_LOW}_attempt${attempt}_$(date +%Y%m%d-%H%M%S)"
        
        # Set CUDA optimizations and PYTHONPATH to ensure 'src' is discoverable
        export PYTHONPATH=$PYTHONPATH:"$PWD"
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
        CERT_PATH=$(python3 -m certifi)
        export SSL_CERT_FILE=${CERT_PATH}
        export REQUESTS_CA_BUNDLE=${CERT_PATH}
        
        # Execute training
        "${TRAIN_CMD[@]}" \
            > >(tee -a "${LOG_PREFIX}.out.log") \
            2> >(tee -a "${LOG_PREFIX}.err.log" >&2)
        
        EXIT_CODE=$?
        
        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "[$(date)] ✅ Range $RANGE_DIR finished successfully." | tee -a "$ERROR_LOG"
            break
        elif [ "$EXIT_CODE" -eq 130 ]; then
            echo "[$(date)] ⏸️  Interrupted by user." | tee -a "$ERROR_LOG"
            exit 130
        fi
        
        echo "[$(date)] ❌ Failed (exit code: $EXIT_CODE). Retrying in ${backoff}s..." | tee -a "$ERROR_LOG"
        sleep "$backoff"
        
        # Exponential backoff
        backoff=$((backoff * BACKOFF_MULTIPLIER))
        if [ "$backoff" -gt "$MAX_BACKOFF" ]; then backoff=$MAX_BACKOFF; fi
    done
    
    # Save progress
    CURRENT_RANGE=$((CURRENT_RANGE + RANGE_STEP))
    echo "$CURRENT_RANGE" > "$CURRENT_RANGE_FILE"
done

echo "[$(date)] 🎉 All ranges complete!" | tee -a "$ERROR_LOG"
rm -f "$CURRENT_RANGE_FILE"
exit 0
