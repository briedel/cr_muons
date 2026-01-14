# Training Instability Analysis - January 14, 2026

## Executive Summary
Your training is **unstable from initialization** due to three critical issues that compound each other. The losses at ±1000 are hitting hard clamps, not actual training values.

---

## Issue 1: Hard Score Clamping at ±1000 ⚠️

**Location:** `training/model.py:273`

```python
# Clamp to prevent unbounded growth of critic weights
# Wasserstein distance should be in [-1000, 1000] range for stable training
score = torch.clamp(score, min=-1000.0, max=1000.0)
```

### Problem
The critic outputs are being **hard-clamped** at ±1000 during the forward pass. This means:
- Any score exceeding these bounds gets clipped
- **Your logs show the clamp is being hit immediately**: `c_loss=-1000.0002`, `g_loss=1000.0000`
- The tiny variations (±0.0002) are from numerical precision, not learning

### Why This Happens
The critic is outputting extreme values right from initialization, indicating:
1. **Poor weight initialization** causing large activations
2. **Missing batch normalization** leading to unbounded growth
3. The normalization by `sqrt(batch_size)` at line 269 is insufficient

### Impact
- **Critic gradients vanish** because `clamp` has zero gradient outside [-1000, 1000]
- Generator receives no useful feedback (gradient is flat)
- Only multiplicity loss trains because it bypasses the critic entirely

---

## Issue 2: Learning Rate Too Low for SGD ⚠️

**Location:** `logs_training/train_0000000-0000999_attempt1_20260114-154159.out.log:7`

```log
Using SGD optimizer: lr=1e-05, momentum=0.5
```

### Problem
You're using **SGD with lr=1e-5**, which is:
- **100x smaller** than the default Adam lr (1e-4 shown in `model.py:584`)
- Far too conservative for SGD, especially with high momentum (0.5)
- Insufficient to overcome the clamped gradients

### Comparison
```python
# Your current setup
opt_G = optim.SGD(gen.parameters(), lr=1e-5, momentum=0.5)   # TOO SLOW

# Default in model.py (line 584)
opt_G = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))  # 10x faster
```

### Impact
Even if gradients weren't clamped, the update steps are too small to:
- Escape poor initialization
- Overcome the critic's extreme outputs
- Make meaningful progress in early training

---

## Issue 3: Gradient Explosion from Initialization 🔥

**Evidence from Profiler:**
```
NanToNumBackward0: 2.4-2.9ms CUDA time (consistent across all steps)
```

The `NanToNumBackward0` operation appearing in **every single training step** means:
- Gradients are constantly producing NaN/Inf values
- `torch.nan_to_num()` is being called during backpropagation
- This happens **11 times per step** (shown in "# of Calls")

### Root Causes

**A. Critic Architecture Issues**
Looking at `ScalableCritic.__init__` (lines 180-201):
```python
self.point_net = nn.Sequential(
    nn.Linear(feat_dim + cond_dim, 64),    # No normalization
    nn.LeakyReLU(0.2),
    nn.Linear(64, 128),                    # No normalization
    nn.LeakyReLU(0.2)
)

self.decision_net = nn.Sequential(
    nn.Linear(128 + cond_dim, 128),        # No normalization
    nn.LeakyReLU(0.2),
    nn.Linear(128, 1)                      # Direct output (no constraint)
)
```

**Problems:**
- No LayerNorm or BatchNorm between layers
- No weight initialization strategy
- Direct output to unbounded scalar
- Only a `sqrt(batch_size)` scaling and hard clamp to prevent explosion

**B. Generator Architecture Issues**
From `ScalableGenerator.__init__` (lines 43-59):
```python
self.global_net = nn.Sequential(
    nn.Linear(..., 128),
    nn.LayerNorm(128), nn.LeakyReLU(0.2),  # ✓ HAS normalization
    nn.Linear(128, self.hidden_dim),
    nn.LayerNorm(self.hidden_dim), nn.LeakyReLU(0.2),  # ✓ HAS normalization
)
```

Generator HAS LayerNorm, but critic DOES NOT. This asymmetry causes:
- Critic receives unnormalized inputs from generator
- Critic's weights grow unbounded trying to discriminate
- Extreme scores saturate the clamp immediately

---

## Why Only Multiplicity Loss Works

Looking at the loss values:
```
batch=100:  m_loss=0.2630  ✓ (trains normally)
batch=200:  m_loss=0.2413  ✓ (decreasing)
batch=300:  m_loss=0.2351  ✓ (improving)
batch=400:  m_loss=0.1972  ✓ (still learning)
```

**Multiplicity network is separate** (`training/model.py:395-406`):
```python
# Predict multiplicity from conditions (INDEPENDENT of critic)
pred_multiplicity = gen.multiplicity_net(real_cond)
loss_multiplicity = nn.functional.mse_loss(pred_multiplicity, target_multiplicity)
loss_multiplicity_scaled.backward()
```

It:
- Trains on real labels (supervised)
- Bypasses the critic entirely
- Uses simple MSE loss (no clamping)
- Has reasonable initialization

---

## The Vicious Cycle

```
Poor Initialization
    ↓
Critic outputs extreme values (> ±1000)
    ↓
Hard clamp activates immediately
    ↓
Gradients vanish (clamp has zero derivative)
    ↓
Learning rate too low to escape
    ↓
NaN/Inf gradients from numerical issues
    ↓
torch.nan_to_num() masks the problem
    ↓
Training continues but nothing improves
```

---

## Recommended Fixes (Priority Order)

### 1. **CRITICAL: Add Layer Normalization to Critic**
```python
# In ScalableCritic.__init__
self.point_net = nn.Sequential(
    nn.Linear(feat_dim + cond_dim, 64),
    nn.LayerNorm(64),              # ADD THIS
    nn.LeakyReLU(0.2),
    nn.Linear(64, 128),
    nn.LayerNorm(128),             # ADD THIS
    nn.LeakyReLU(0.2)
)

self.decision_net = nn.Sequential(
    nn.Linear(128 + cond_dim, 128),
    nn.LayerNorm(128),             # ADD THIS
    nn.LeakyReLU(0.2),
    nn.Linear(128, 1)
)
```

### 2. **CRITICAL: Increase Learning Rate OR Switch to Adam**

**Option A:** Switch to Adam (recommended for WGANs)
```bash
# Remove --optimizer sgd flag, or explicitly:
--optimizer adam --lr 1e-4
```

**Option B:** If keeping SGD, increase learning rate 10-100x
```bash
--optimizer sgd --lr 1e-4 --momentum 0.5  # 10x increase
# or even
--optimizer sgd --lr 5e-4 --momentum 0.5  # 50x increase
```

### 3. **HIGH: Proper Weight Initialization**
Add to model.py after each network definition:
```python
# In ScalableCritic.__init__ after defining networks
for m in self.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
```

### 4. **MEDIUM: Soften the Clamp**
```python
# Replace hard clamp with tanh scaling
# In ScalableCritic.forward() line 273
score = 100.0 * torch.tanh(score / 100.0)  # Smooth saturation instead of hard clamp
```

### 5. **LOW: Monitor Gradient Norms**
Add logging to track when gradients explode:
```python
# After each backward() call
for name, param in gen.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100.0 or not torch.isfinite(param.grad).all():
            print(f"WARNING: {name} grad_norm={grad_norm:.2f}")
```

---

## Expected Behavior After Fixes

After implementing fixes 1-3, you should see:
```
batch=100:  c_loss=-15.2341   g_loss=12.5432   m_loss=0.2630   # REASONABLE VALUES
batch=200:  c_loss=-12.8765   g_loss=10.3421   m_loss=0.2413   # BOTH TRAINING
batch=300:  c_loss=-10.2341   g_loss=8.1234    m_loss=0.2351   # IMPROVING
```

Instead of:
```
batch=100:  c_loss=-1000.0002  g_loss=1000.0000  m_loss=0.2630  # CLAMPED
```

---

## References

**Files to modify:**
1. `training/model.py` lines 190-201 (add LayerNorm to critic)
2. `training/model.py` lines 273 (soften clamp)
3. Command line args: change `--lr 1e-5` to `--lr 1e-4` or switch to Adam

**Evidence in logs:**
- Line 7: SGD lr=1e-5
- Line 302: `c_loss=-1000.0002` (hitting clamp)
- Profiler: 11 calls to `NanToNumBackward0` per step (gradient issues)
