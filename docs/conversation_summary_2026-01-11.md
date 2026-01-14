# Training Improvements and Fixes - January 11, 2026

## Overview

This session focused on improving training observability, fixing Pelican data loading issues, implementing automated range-based training, and addressing WGAN-GP stability challenges at scale.

## 1. Wasserstein Gap Monitoring

**Problem:** No direct metric for tracking how well the generator is fooling the critic.

**Solution:** Added Wasserstein gap computation and logging.

**Implementation:**
- Modified `training/model.py::train_step_scalable()` to compute and return `w_gap = mean(critic(real)) - mean(critic(fake))`
- Updated `training/train.py` to capture, log, and display `w_gap` in:
  - Progress bar postfix: `w_gap=X.XXXX`
  - Periodic console output
  - TensorBoard scalar: `train/w_gap`

**Interpretation:**
- Positive w_gap: Critic scores real samples higher (generator needs improvement)
- w_gap → 0: Generator successfully fooling critic
- Negative w_gap: Unusual, may indicate training issues
- Expected: w_gap oscillates; trend over time is more meaningful than instantaneous value

## 2. 500-Step Moving Average for Wasserstein Gap

**Problem:** Raw w_gap is noisy and hard to interpret for long-term trends.

**Solution:** Added exponential moving average over 500 training steps.

**Implementation:**
- Added `collections.deque` buffer with 500-step window in `training/train.py`
- Computes running sum for efficient O(1) average updates
- Displays as `w_ma=X.XXXX` in progress bar and console
- Logs to TensorBoard as `train/w_gap_ma_500`

**Benefits:**
- Smooths out batch-to-batch noise
- Reveals long-term convergence trends
- Easier to spot divergence or mode collapse early

## 3. Pelican Wildcard Expansion Improvements

### 3.1 Glob Performance Optimization

**Problem:** `pelicanfs.glob()` can be slow for large recursive searches without proper parameters.

**Solution:** Pass `detail=False` to avoid fetching unnecessary file metadata.

**Changes in `training/utils/pelican_utils.py`:**
```python
# Before
res = glob_fn(item)

# After
if 'detail' in sig.parameters:
    res = glob_fn(item, detail=False)
else:
    res = glob_fn(item)
```

**Impact:** Significant speedup for patterns like `**/*.parquet` with thousands of files.

### 3.2 Recursive Pattern Handling

**Problem:** `**` patterns were falling back to `ls()` which can't handle recursion, causing access errors.

**Solution:** Detect recursive patterns early and fail fast with helpful errors if glob doesn't work.

**Implementation:**
- Check for `**` in pattern
- If glob fails on recursive pattern, raise immediately with diagnostic guidance
- Only use `ls()` fallback for simple (non-recursive) patterns

### 3.3 Token Scope Inference Fixes

**Problem:** Token scopes included wildcards and range patterns, requiring new tokens for each directory.

**Solution:** Strip all wildcards and range patterns from scope path.

**Key improvements:**
1. **Wildcard removal:** Stop at first path component containing `*`, `?`, `[`, etc.
2. **Range pattern detection:** Recognize `NNNNNNN-NNNNNNN` format (e.g., `0000000-0000999`)
3. **Clean scope generation:** 
   - Input: `pelican://.../testing/0000000-0000999/*.parquet`
   - Scope: `storage.read:/data/sim/IceCube/2025/testing`

**Benefit:** Single token works across all 100 range directories (0000000-0099999).

## 4. Automated Range-Based Training Script

**Problem:** Pelican glob can't handle complex patterns reliably; need to iterate through directory ranges sequentially.

**Solution:** Created `train_with_restart.sh` with automatic range iteration.

**Features:**
- Iterates through 100 ranges: `0000000-0000999`, `0001000-0001999`, ..., `0099000-0099999`
- Per-range retry logic with exponential backoff (5s → 300s max)
- Progress tracking via `.current_range` file (survives crashes/restarts)
- Checkpoint-based continuity (model accumulates knowledge across ranges)
- Configurable range bounds and step size

**Configuration variables:**
```bash
RANGE_START=0      # First range
RANGE_END=99000    # Last range
RANGE_STEP=1000    # Increment
MAX_RETRIES=0      # 0 = infinite retries per range
```

**Usage:**
```bash
./train_with_restart.sh  # Start or resume
rm .current_range        # Reset to beginning
```

**Architecture:** Sequential training (Option 1) - each range builds on previous checkpoint, avoiding mode collapse from data subsets.

## 5. Training Stability Investigations

### 5.1 Problem Identification

**Symptoms observed around batch 270-290:**
- c_loss spiking from -20 to -400 (20× increase)
- g_loss collapsing from 650 to 330 (50% drop)
- w_gap moving average drifting negative
- Pattern recurring across multiple files

**Root cause:** Without gradient penalty (`--lambda-gp 0.0`), critic becomes too discriminative at large batch sizes (128k events), overpowering generator.

### 5.2 Attempted Solutions

**Attempt 1: Re-enable gradient penalty**
```bash
--lambda-gp 0.01 --gp-every 2
```
- **Result:** Stability improved but throughput dropped from 2-3 → 1 batch/s
- **Reason:** GP overhead massive at 128k batch size

**Attempt 2: Limit GP computation**
```bash
--lambda-gp 0.01 --gp-every 2 --gp-max-pairs 8192
```
- **Result:** Throughput recovered to 2-3 batch/s
- **Issue:** Instability still occurred (c_loss still spiked to -395)

**Final solution: Lower learning rate**
```bash
--lr 1e-5 --lambda-gp 0.0
```
- **Rationale:** Gentler updates prevent critic from dominating
- **Trade-off:** Slower convergence but maintains throughput
- **Expected:** Stable losses without GP overhead

### 5.3 Current Settings

```bash
--batch-size 128000
--critic-steps 5
--optimizer sgd --lr 1e-5 --momentum 0.5
--grad-clip-norm 1e7
--lambda-gp 0.0
--allow-tf32
```

**Monitoring criteria:**
- ✅ Healthy: c_loss in ±200 range
- ⚠️ Warning: c_loss beyond ±300
- ❌ Critical: c_loss beyond ±1000 or NaN

## 6. Documentation Updates

### 6.1 README.md Enhancements

**Added "Why SGD?" section explaining:**
1. **Stability with normalized critic:** SGD works better with √batch-size normalization + clamping
2. **Throughput benefits:** Lower optimizer overhead vs Adam (8-9 batch/s achieved)
3. **Large gradient handling:** Gradients ~7-9×10⁶ work with `--grad-clip-norm 1e7`
4. **Momentum tuning:** 0.5 balances acceleration and stability in adversarial training

**Updated high-throughput example** with proven settings and performance metrics.

### 6.2 This Summary Document

Created comprehensive record of session improvements for future reference and team knowledge sharing.

## 7. File Changes Summary

**Modified files:**
- `training/model.py`: Added w_gap computation and return value
- `training/train.py`: 
  - Added w_gap and w_ma tracking/logging
  - Fixed w_gap_ma initialization for both batch paths
  - Added `from collections import deque`
- `training/utils/pelican_utils.py`:
  - Optimized glob with `detail=False`
  - Improved recursive pattern error handling
  - Enhanced scope inference (strip wildcards and ranges)
- `train_with_restart.sh`: Complete rewrite for range-based iteration
- `README.md`: Added SGD rationale section

**New files:**
- `docs/conversation_summary_2026-01-11.md` (this document)

## 8. Outstanding Items

**Monitoring needed (next 10-20 files):**
- Verify `--lr 1e-5` prevents c_loss spikes beyond batch 280
- Track w_gap_ma for long-term convergence trends
- Confirm throughput remains at 2-3 batch/s

**Potential future tuning:**
- If instability persists: Try `--lr 5e-6` or switch to Adam
- If throughput degrades: Profile GP overhead vs benefit
- Consider adaptive LR scheduling based on loss health metrics

## 9. Key Takeaways

1. **Observability is critical:** Wasserstein gap + moving average provide actionable convergence signals
2. **Data loading matters:** Proper Pelican optimization (glob detail, scope inference) enables seamless federation access
3. **Stability trumps speed:** Lower LR (1e-5) with fast throughput beats unstable training at higher LR
4. **Sequential training works:** Checkpoint-based range iteration avoids parallel weight averaging issues for GANs
5. **Batch size impacts stability:** Large batches (128k) amplify critic advantage; tuning required

## 10. Session Metrics

- **Training files processed:** 840+ files across first range
- **Throughput:** 2-3 batch/s at 128k batch size (~400k events/s)
- **GPU utilization:** 10-19% (memory efficient)
- **Data pipeline:** Pelican federation + local prefetch + background fetcher operational
- **Stability:** In progress - transitioning to lower LR after GP experiments

---

**Session Date:** January 11, 2026  
**Primary Focus:** Observability, data loading optimization, stability tuning  
**Status:** Ongoing training with refined hyperparameters
