#!/bin/bash
# Robust training launcher with automatic restart on failure
# Handles Pelican federation connection issues gracefully
# Iterates through directory ranges (0000000-0000999, 0001000-0001999, etc.)

set -o pipefail

# Configuration
MAX_RETRIES=0  # 0 = infinite retries
INITIAL_BACKOFF=5  # seconds
MAX_BACKOFF=300  # seconds (5 minutes)
BACKOFF_MULTIPLIER=2

# Range configuration for directory iteration
RANGE_START=0      # Starting range (0000000)
RANGE_END=99000    # Ending range (0099000)
RANGE_STEP=1000    # Increment by 1000
CURRENT_RANGE_FILE=".current_range"
# Limit processing to the next N ranges from current position (set 0 to use RANGE_END)
RANGES_TO_PROCESS=2

# Logging
LOG_DIR="./logs_tensorboard"
ERROR_LOG="${LOG_DIR}/training_errors.log"
RUN_LOG_DIR="./logs_training"
mkdir -p "$LOG_DIR"
mkdir -p "$RUN_LOG_DIR"

# Load current range from file, or start at beginning
if [ -f "$CURRENT_RANGE_FILE" ]; then
    CURRENT_RANGE=$(cat "$CURRENT_RANGE_FILE")
else
    CURRENT_RANGE=$RANGE_START
fi

# If limiting to the next N ranges, tighten RANGE_END for this run
if [ "$RANGES_TO_PROCESS" -gt 0 ]; then
    RANGE_END=$(( CURRENT_RANGE + RANGE_STEP * (RANGES_TO_PROCESS - 1) ))
fi

echo "=== Training Launcher with Auto-Restart ===" | tee -a "$ERROR_LOG"
echo "Start time: $(date)" | tee -a "$ERROR_LOG"
echo "Max retries: $MAX_RETRIES (0 = infinite)" | tee -a "$ERROR_LOG"
echo "Initial backoff: ${INITIAL_BACKOFF}s" | tee -a "$ERROR_LOG"
echo "Range: ${RANGE_START} to ${RANGE_END} (step: ${RANGE_STEP})" | tee -a "$ERROR_LOG"
echo "" | tee -a "$ERROR_LOG"

# Iterate through ranges
while [ "$CURRENT_RANGE" -le "$RANGE_END" ]; do
    # Format range with leading zeros (7 digits)
    RANGE_LOW=$(printf "%07d" $CURRENT_RANGE)
    RANGE_HIGH=$(printf "%07d" $((CURRENT_RANGE + RANGE_STEP - 1)))
    RANGE_DIR="${RANGE_LOW}-${RANGE_HIGH}"
    
    echo "[$(date)] 📁 Processing range: $RANGE_DIR" | tee -a "$ERROR_LOG"
    
    # Build training command with current range
    TRAIN_CMD=(
        stdbuf -oL -eL python -u training/train.py \
        -i "pelican://osg-htc.org/icecube/wipac/data/sim/IceCube/2025/testing/${RANGE_DIR}/*.parquet" \
        --use-hf --parquet-batch-reader \
        --prefetch-batches 10 --num-workers 4 --prefetch-factor 4 --pin-memory \
        --batch-size 128000 --log-interval 100 --max-muons-per-event 200000 --max-muons-per-batch 30000000 \
        --drop-empty-events \
        --multi-file-shuffle 10 \
        --memory-cache-mb 2048 \
        --profile-steps 10 \
        --tb-logdir ./logs_tensorboard/ --tb-hist-interval 10 \
        --prefetch-dir ./testdata/ --prefetch-delete-after-use --prefetch-max-files 500 --prefetch-ahead 20 \
        --auto-token --checkpoint ./training_checkpoint.json --model-checkpoint ./model_checkpoint.pt \
        --checkpoint-io local --device cuda \
        --optimizer adam --lr 1e-4 --grad-clip-norm 0.0 \
        --allow-tf32 --lambda-gp 10.0 --gp-max-pairs 4096 --gp-every 2 \
        --adaptive-critic --critic-steps 2 \
        --w-ma-low -5.0 --w-ma-high 10.0 \
        --critic-steps-min 1 --critic-steps-max 3 \
        --lambda-gp-min 10.0 --lambda-gp-max 20.0 \
        --gp-adapt-factor 1.5
    )
    
    # Retry loop for current range
    attempt=0
    backoff=$INITIAL_BACKOFF
    range_success=false
    
    while true; do
        attempt=$((attempt + 1))
        
        # Check retry limit
        if [ "$MAX_RETRIES" -gt 0 ] && [ "$attempt" -gt "$MAX_RETRIES" ]; then
            echo "[$(date)] ❌ Max retries ($MAX_RETRIES) exceeded for range $RANGE_DIR. Moving to next range." | tee -a "$ERROR_LOG"
            break
        fi
        
        echo "[$(date)] 🚀 Range $RANGE_DIR - Attempt $attempt (backoff: ${backoff}s)" | tee -a "$ERROR_LOG"
        echo "[$(date)] Starting training..." | tee -a "$ERROR_LOG"
        
        # Run training with stdout/stderr tee'd to per-range logs while streaming to screen
        LOG_PREFIX="${RUN_LOG_DIR}/train_${RANGE_LOW}-${RANGE_HIGH}_attempt${attempt}_$(date +%Y%m%d-%H%M%S)"
        "${TRAIN_CMD[@]}" \
            > >(tee -a "${LOG_PREFIX}.out.log") \
            2> >(tee -a "${LOG_PREFIX}.err.log" >&2)
        EXIT_CODE=$?
        
        # Check if training succeeded (exit 0) or was interrupted by user (Ctrl+C = 130)
        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "[$(date)] ✅ Range $RANGE_DIR completed successfully." | tee -a "$ERROR_LOG"
            range_success=true
            break
        elif [ "$EXIT_CODE" -eq 130 ]; then
            echo "[$(date)] ⏸️  Training interrupted by user (Ctrl+C)." | tee -a "$ERROR_LOG"
            exit 130
        fi
        
        # Training failed—log and retry
        echo "[$(date)] ❌ Training failed (exit code: $EXIT_CODE)" | tee -a "$ERROR_LOG"
        echo "[$(date)] 💤 Waiting ${backoff}s before retry..." | tee -a "$ERROR_LOG"
        
        # Sleep before retry
        sleep "$backoff"
        
        # Increase backoff exponentially (cap at MAX_BACKOFF)
        backoff=$((backoff * BACKOFF_MULTIPLIER))
        if [ "$backoff" -gt "$MAX_BACKOFF" ]; then
            backoff=$MAX_BACKOFF
        fi
        
        echo "" | tee -a "$ERROR_LOG"
    done
    
    # Move to next range
    CURRENT_RANGE=$((CURRENT_RANGE + RANGE_STEP))
    echo "$CURRENT_RANGE" > "$CURRENT_RANGE_FILE"
    
    if [ "$CURRENT_RANGE" -le "$RANGE_END" ]; then
        echo "[$(date)] ➡️  Moving to next range..." | tee -a "$ERROR_LOG"
        echo "" | tee -a "$ERROR_LOG"
    fi
done

echo "[$(date)] 🎉 All ranges completed (${RANGE_START} to ${RANGE_END})!" | tee -a "$ERROR_LOG"
rm -f "$CURRENT_RANGE_FILE"
exit 0
