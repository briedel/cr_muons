# Adaptive Critic Tuning & Empty Event Filtering Fixes

**Date:** January 12, 2026  
**Training Range:** 0000000-0000999 (1000 files)  
**Checkpoint Status:** Resumed from 924/1000 files processed

## Problem Summary

### 1. Oscillatory Training Dynamics
- **Issue:** w_ma exhibited boom-bust cycles at batches 3400, 6300, 9200, 11200
- **Root Cause:** Bidirectional adaptive critic tuning created limit cycles
  - System weakened critic when w_ma < -5.0 (reduced critic_steps, increased lambda_gp)
  - System strengthened critic when w_ma > +10.0 (increased critic_steps, decreased lambda_gp)
  - Created oscillatory behavior: weaken → recover → strengthen → collapse → repeat

### 2. Throughput Decline
- **Observation:** Batch processing rate dropped from 2.75-3 batches/s to 1.75 batches/s
- **Initial Suspect:** --drop-empty-events flag overhead
- **Actual Causes:**
  1. Aggressive prefetch settings (1000/6/6) caused I/O contention with remote Pelican storage
  2. Data corruption from broken filtering logic (see below)

### 3. Non-Functional Empty Event Filtering
- **Issue:** --drop-empty-events flag present but not filtering
- **Evidence:** All logs showed skipped_empty=0 despite flag being enabled
- **Cause:** Flag was added to command line but filtering logic not implemented

### 4. Critical Data Corruption Bug
- **Symptom:** Impossible statistics - events=358M muons=181M (should be muons >> events)
- **Root Cause:** First filtering implementation removed empty events but didn't filter corresponding muons
- **Impact:** Orphaned muons pointing to non-existent primaries, data corruption, processing inefficiency

## Solutions Implemented

### 1. Unidirectional Adaptive Critic Tuning
**Files Modified:** [training/train.py](../training/train.py) (lines ~803-831, ~1367-1395)

**Changes:**
- Removed `elif w_gap_ma > w_high` branch (upward reversal)
- System now only weakens critic (never strengthens) when w_ma < -5.0
- Removed unused variables: `w_high`, `cs_max`, `gp_min`, `lambda_gp_min`
- Added comments: "UNIDIRECTIONAL: Only weaken critic, never strengthen (prevents oscillations)"

**Logic:**
```python
if w_gap_ma < w_low:  # w_low = -5.0
    critic_steps = max(1, critic_steps - 1)  # min: 1
    lambda_gp = min(20.0, lambda_gp * 1.5)   # max: 20.0
# No upward reversal - prevents oscillations
```

**Expected Outcome:** Prevents oscillatory cycles; system stays at weakened settings once triggered

### 2. Prefetch Optimization
**File Modified:** train_with_restart.sh (line 63)

**Changes:**
- `--prefetch-batches`: 1000 → 2
- `--num-workers`: 6 → 1
- `--prefetch-factor`: 6 → 2
- Removed `--persistent-workers`

**Rationale:** Conservative settings reduce I/O contention with remote Pelican storage

### 3. Microbatch Size Increase
**File Modified:** train_with_restart.sh (line 63)

**Changes:**
- `--max-muons-per-batch`: 5000000 → 10000000

**Rationale:** Reduces overhead when VRAM permits (current usage 2.5G/8G shows headroom)

### 4. Empty Event Filtering Implementation & Bug Fix
**Files Modified:** [training/train.py](../training/train.py) (lines ~630-650, ~1070-1090)

**Initial Implementation (Broken):**
```python
keep_mask = counts_cpu > 0
counts = counts[keep_mask]
prims = prims[keep_mask]
# Remap batch_idx with cumsum
new_indices = torch.arange(keep_mask.sum())
old_to_new = torch.zeros(keep_mask.shape[0], dtype=torch.long)
old_to_new[keep_mask] = new_indices
batch_idx = old_to_new[batch_idx]  # BUG: didn't filter muons first
```
**Problem:** Remapped batch_idx without filtering muons → orphaned muons pointing to removed events

**Fixed Implementation:**
```python
keep_mask = counts_cpu > 0
counts = counts[keep_mask]
prims = prims[keep_mask]

# Create old-to-new index mapping
old_to_new = torch.full((keep_mask.shape[0],), -1, dtype=torch.long, device=batch_idx.device)
old_to_new[keep_mask] = torch.arange(keep_mask.sum(), device=batch_idx.device)

# Filter muons belonging to kept events
muons_keep = old_to_new[batch_idx] >= 0
real_muons = real_muons[muons_keep]
batch_idx = old_to_new[batch_idx[muons_keep]]

# Update statistics
skipped_empty += (keep_mask.shape[0] - keep_mask.sum().item())
```

**Key Fix:** Explicitly filter muons before remapping batch_idx using old_to_new mapping

## Data Model Clarification

- **events** = cosmic ray primaries (parent particles)
- **muons** = daughter muons produced by primaries
- **Normal relationship:** muons >> events (each primary produces multiple muons)
- **Example:** events=100M muons=400M → ~4 muons/primary average
- **Impossible:** events > muons (indicates data corruption)

## Current Configuration

### Adaptive Tuning Parameters
- `--adaptive-critic` enabled
- `--critic-steps 2` (base, min 1, max unused)
- `--w-ma-low -5.0` (trigger threshold)
- `--w-ma-high 10.0` (unused after unidirectional change)
- `--lambda-gp 10.0` (base, min unused, max 20.0)
- `--gp-adapt-factor 1.5`

### Data Loading
- `--use-hf --parquet-batch-reader`
- `--multi-file-shuffle 10` (concurrent files)
- `--prefetch-batches 2`
- `--num-workers 1`
- `--prefetch-factor 2`
- `--pin-memory`

### Microbatching
- `--max-muons-per-batch 10000000`
- `--max-muons-per-event 200000`

### Empty Event Filtering
- `--drop-empty-events` enabled (now functional)

### Training
- Optimizer: SGD lr=1e-5, momentum=0.5
- Gradient clipping: 1e7
- Gradient penalty: λ=10.0 (base), max_pairs=4096, every=2 batches

## Validation Checklist

After next training run, verify:
- [ ] `skipped_empty > 0` appears in logs when empty events present
- [ ] `events < muons` consistently (e.g., events=100M muons=400M)
- [ ] Batch processing rate recovers toward 2.75-3 batches/s
- [ ] No oscillatory w_ma patterns (unidirectional adaptation prevents reversals)
- [ ] TensorBoard adapt/* scalars show stable weakened settings after trigger

## Future Optimization Options

1. **Pre-filter data:** Generate parquet files with empty events already removed to eliminate runtime overhead
2. **Increase microbatch size:** Try 20-30M muons (VRAM usage 2.5G/8G shows significant headroom)
3. **Monitor adaptation frequency:** Compare TensorBoard adapt/* scalars across runs to validate unidirectional strategy reduces adaptation vs bidirectional

## Files Modified

1. [training/train.py](../training/train.py)
   - Lines ~803-831, ~1367-1395: Unidirectional adaptive tuning
   - Lines ~630-650: Multi-file path empty event filtering fix
   - Lines ~1070-1090: Sequential path empty event filtering fix

2. train_with_restart.sh
   - Line 63: Prefetch settings, microbatch size, adaptive flags

## Timeline

- **Issue Identified:** Oscillatory w_ma pattern in training logs
- **Unidirectional Strategy:** Implemented to prevent limit cycles
- **Throughput Decline:** Investigated correlation with --drop-empty-events
- **Filtering Bug:** Discovered flag not functional (skipped_empty=0)
- **Initial Fix:** Implemented filtering but introduced data corruption
- **Critical Bug:** Diagnosed impossible events > muons ratio
- **Final Fix:** Proper muon filtering with old_to_new index mapping
- **Status:** Fixed code deployed, awaiting validation in new training run
