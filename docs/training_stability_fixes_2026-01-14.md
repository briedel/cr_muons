# Training Stability Fixes - January 14, 2026

## Problem Summary

Training was unstable from initialization with losses stuck at ±1000, indicating gradient death and numerical saturation. Only the multiplicity loss was training normally.

**Symptoms:**
- `c_loss=-1000.0002, g_loss=1000.0000` (saturated from batch 1)
- `NanToNumBackward0` appearing 11 times per training step (~2.7ms CUDA overhead)
- Only `m_loss` showed normal training behavior (0.263 → 0.197)

**Root Causes:**
1. Missing LayerNorm in ScalableCritic causing unbounded activations
2. Hard clamp at ±1000 killing gradients outside bounds
3. Poor weight initialization leading to extreme initial outputs
4. Learning rate 100x too small (SGD lr=1e-5 instead of Adam lr=1e-4)

---

## Changes Implemented

### 1. Added LayerNorm to ScalableCritic Architecture

**File:** `training/model.py` (lines 200-215)

**Changes:**
```python
# BEFORE: No normalization layers
self.point_net = nn.Sequential(
    nn.Linear(feat_dim + cond_dim, 64),
    nn.LeakyReLU(0.2),
    nn.Linear(64, 128),
    nn.LeakyReLU(0.2)
)

# AFTER: Added LayerNorm after each Linear layer
self.point_net = nn.Sequential(
    nn.Linear(feat_dim + cond_dim, 64),
    nn.LayerNorm(64),              # NEW
    nn.LeakyReLU(0.2),
    nn.Linear(64, 128),
    nn.LayerNorm(128),             # NEW
    nn.LeakyReLU(0.2)
)

self.decision_net = nn.Sequential(
    nn.Linear(128 + cond_dim, 128),
    nn.LayerNorm(128),             # NEW
    nn.LeakyReLU(0.2),
    nn.Linear(128, 1)
)
```

**Impact:** Prevents unbounded activation growth and matches the normalization already present in ScalableGenerator.

---

### 2. Added Xavier Weight Initialization

**File:** `training/model.py` (lines 65-71, 220-226)

**Changes:**
- Added `_initialize_weights()` method to both `ScalableGenerator` and `ScalableCritic`
- Applied Xavier normal initialization with `gain=0.5` for conservative scaling
- Called from `__init__()` after network construction

```python
def _initialize_weights(self):
    """Initialize weights conservatively for stable training"""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

**Impact:** Prevents extreme initial outputs that were saturating the clamp immediately.

---

### 3. Replaced Hard Clamp with Smooth Saturation

**File:** `training/model.py` (line 298)

**Change:**
```python
# BEFORE: Hard clamp with zero gradient outside bounds
score = torch.clamp(score, min=-1000.0, max=1000.0)

# AFTER: Smooth tanh saturation with gradients everywhere
score = 100.0 * torch.tanh(score / 100.0)
```

**Impact:** 
- Preserves gradient flow even when scores are large
- Smooth boundaries instead of abrupt cutoff
- Output range approximately [-100, +100] with smooth approach

---

### 4. Updated Optimizer Configuration

**File:** `train_with_restart.sh` (line 72)

**Change:**
```bash
# BEFORE: SGD with 100x too small learning rate
--optimizer sgd --lr 1e-5 --momentum 0.5 --grad-clip-norm 1e7

# AFTER: Adam with standard learning rate
--optimizer adam --lr 1e-4 --grad-clip-norm 0.0
```

**Impact:** Faster convergence and better handling of sparse gradients in WGAN-GP training.

---

## Results

**Before fixes:**
```
batch=100:  c_loss=-1000.0002  g_loss=1000.0000   m_loss=0.2630  (SATURATED)
batch=200:  c_loss=-1000.0002  g_loss=1000.0000   m_loss=0.2413  (NO LEARNING)
```

**After fixes:**
```
batch=100:  c_loss=0.0217   g_loss=-0.0076   m_loss=0.0368  (TRAINING)
batch=200:  c_loss=0.0398   g_loss=-0.0160   m_loss=0.0266  (IMPROVING)
```

**Key improvements:**
- ✅ All three losses now training in reasonable range
- ✅ No more saturation at ±1000
- ✅ Critic and generator both receiving proper gradients
- ⚠️ `NanToNumBackward0` still present (defensive sanitization in 6 locations)

---

## Remaining Optimizations (Optional)

The `NanToNumBackward0` operations (11 calls per step, ~2.7ms CUDA) suggest defensive `torch.nan_to_num()` calls that could be removed if NaNs don't actually occur:

**Locations in `training/model.py`:**
- Line 88, 91: Multiplicity head sanitization
- Line 145: Generator count decoding
- Line 248: Critic point features
- Line 286: Critic global features  
- Line 354: Gradient penalty computation

**Next steps (if needed):**
1. Monitor for actual NaN/Inf occurrences during training
2. Remove sanitization calls if no issues arise
3. Potential speedup: ~2.7ms per training step

---

## Files Modified

1. `training/model.py` - Core architecture and initialization fixes
2. `train_with_restart.sh` - Optimizer configuration update

## Related Documentation

- `docs/INSTABILITY_ANALYSIS.md` - Detailed root cause analysis
- Training logs: `logs_training/train_0000000-0000999_attempt1_20260114-154847.out.log`

---

**Summary:** Three critical fixes (LayerNorm, weight init, smooth saturation) plus optimizer tuning resolved the training instability. Losses now train in expected range for WGAN-GP instead of saturating at ±1000.
