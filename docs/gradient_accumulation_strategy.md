# Gradient Accumulation Strategy for VRAM Optimization

**Date**: January 17, 2026  
**Status**: Proposed Solution  
**Context**: Addressing CUDA OOM errors during GAN training with variable-sized cosmic ray muon batches

## Problem Statement

Training is experiencing CUDA out-of-memory (OOM) errors on a 7.66 GiB GPU when processing large batches:
- Memory usage grows from 3-5% early in training to 11-12% at ~3700-3800 batches
- Crashes occur when attempting to allocate 400-500 MiB with insufficient free memory
- Large batches contain up to 690k muons despite preflight splitting at 420k threshold
- `max_alloc` progressively grows across attempts (4960 MiB → 7178 MiB)

Current mitigation strategies (cache clearing, optimizer state cleanup) help but don't fully prevent OOM.

## Gradient Accumulation Overview

**Gradient accumulation** allows training with effective large batch sizes while processing smaller sub-batches sequentially. Instead of:
1. Process full batch → compute gradients → update weights

We do:
1. Process sub-batch 1 → accumulate gradients (no weight update)
2. Process sub-batch 2 → accumulate gradients (no weight update)
3. ...
4. Process sub-batch N → accumulate gradients → **now** update weights

### Key Benefits
- **Reduced peak VRAM**: Only one sub-batch in memory at a time
- **Maintains effective batch size**: Statistical properties preserved
- **No architecture changes**: Works with existing GAN structure
- **Deterministic**: Given same data order, produces identical results

### Trade-offs
- **Slower training**: More forward/backward passes per weight update
- **Complexity**: Requires careful implementation for GAN dynamics
- **Batch statistics**: BatchNorm/LayerNorm may behave differently (not applicable here)

## Implementation Considerations for WGAN-GP

### Challenge: Critic vs Generator Dynamics

Our training loop has asymmetric updates:
```python
for each batch:
    for critic_steps (typically 2-3):
        update_critic(batch)
    update_generator(batch)
```

With gradient accumulation, we need to decide:
1. **Accumulate within critic steps?** (simpler but changes dynamics)
2. **Accumulate across full cycles?** (complex but preserves dynamics)
3. **Hybrid approach?** (accumulate only when needed)

### Recommended Approach: Dynamic Accumulation

Only enable gradient accumulation when batch size exceeds a threshold:

```python
# Configuration
ACCUMULATION_THRESHOLD = 250000  # muons
ACCUMULATION_STEPS = 2           # split into N sub-batches

def should_accumulate(batch):
    total_muons = batch.sum_muons()
    return total_muons > ACCUMULATION_THRESHOLD

def train_step_with_accumulation(batch, model, optimizer):
    if should_accumulate(batch):
        # Split batch into sub-batches
        sub_batches = split_batch(batch, n_splits=ACCUMULATION_STEPS)
        
        # Accumulate gradients
        for i, sub_batch in enumerate(sub_batches):
            loss = compute_loss(model, sub_batch)
            # Scale loss by number of accumulation steps
            (loss / ACCUMULATION_STEPS).backward()
        
        # Single optimizer step after all sub-batches
        optimizer.step()
        optimizer.zero_grad()
    else:
        # Standard training for normal batches
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Implementation Details

#### 1. Batch Splitting Strategy

For our cosmic ray data:
```python
def split_batch_by_events(muon_features, primary_features, n_splits):
    """
    Split batch while preserving event boundaries.
    
    Args:
        muon_features: (total_muons, 3) tensor
        primary_features: (num_events, 4) tensor
        n_splits: number of sub-batches
    
    Returns:
        List of (muon_sub_batch, primary_sub_batch) tuples
    """
    num_events = primary_features.size(0)
    events_per_split = num_events // n_splits
    
    sub_batches = []
    muon_offset = 0
    
    for i in range(n_splits):
        start_event = i * events_per_split
        end_event = (i + 1) * events_per_split if i < n_splits - 1 else num_events
        
        # Get primary features for this slice
        primary_sub = primary_features[start_event:end_event]
        
        # Count muons in these events to slice muon_features
        muons_in_slice = count_muons_in_events(start_event, end_event)
        muon_sub = muon_features[muon_offset:muon_offset + muons_in_slice]
        muon_offset += muons_in_slice
        
        sub_batches.append((muon_sub, primary_sub))
    
    return sub_batches
```

#### 2. Loss Scaling

Critical for correct gradient magnitudes:
```python
# WRONG - gradients too small
loss.backward()  # each sub-batch contributes full loss

# CORRECT - average across accumulation steps
(loss / n_accumulation_steps).backward()
```

#### 3. Gradient Penalty Considerations

WGAN-GP computes gradient penalty on interpolated samples:
```python
def gradient_penalty_with_accumulation(critic, real, fake, n_accumulation):
    """
    Gradient penalty respects accumulation.
    """
    # Sample interpolation points
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    
    # Critic forward pass
    d_interpolates = critic(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Gradient penalty
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    # Scale by accumulation steps
    return gp / n_accumulation
```

#### 4. Monitoring Memory Savings

Add logging to track effectiveness:
```python
if accumulated:
    tqdm.write(
        f"[accumulation] batch split into {n_splits} sub-batches: "
        f"total_muons={total_muons} max_sub_muons={max_sub_muons} "
        f"peak_mem={peak_allocated_mib}MiB"
    )
```

## Configuration Parameters

Add to `train.py` argument parser:
```python
parser.add_argument(
    "--gradient-accumulation-steps",
    type=int,
    default=1,
    help="Number of sub-batches for gradient accumulation (1=disabled)"
)
parser.add_argument(
    "--accumulation-muon-threshold",
    type=int,
    default=0,
    help="Only accumulate when batch exceeds this many muons (0=always accumulate)"
)
```

Usage in training script:
```bash
python training/train.py \
    --gradient-accumulation-steps 2 \
    --accumulation-muon-threshold 250000 \
    # ... other flags
```

## Alternative: Adaptive Accumulation

More sophisticated approach that adjusts accumulation based on available memory:

```python
def adaptive_accumulation_steps(batch, target_memory_mib=2048):
    """
    Determine accumulation steps needed to fit in target memory.
    """
    estimated_memory = estimate_batch_memory(batch)
    
    if estimated_memory <= target_memory_mib:
        return 1  # No accumulation needed
    
    # Calculate required splits
    n_splits = math.ceil(estimated_memory / target_memory_mib)
    return min(n_splits, 4)  # Cap at 4x accumulation
```

## Integration with Existing Memory Management

Gradient accumulation complements current strategies:

| Strategy | When | Purpose |
|----------|------|---------|
| **Preflight splitting** | Before batch creation | Prevent catastrophically large batches |
| **Gradient accumulation** | During training | Handle moderately large batches gracefully |
| **Cache clearing** | Every 1000 steps | Reclaim fragmented memory |
| **Optimizer state cleanup** | Every 1000 steps | Remove stale momentum buffers |

Recommended settings with gradient accumulation:
```bash
--preflight-muon-threshold 350000        # Relaxed from 420k
--accumulation-muon-threshold 250000     # Accumulate for large batches
--gradient-accumulation-steps 2          # 2x splitting when needed
--cuda-empty-cache-interval 500          # More frequent clearing
--cuda-empty-cache-threshold-mib 1536    # Lower threshold
```

## Testing Plan

1. **Baseline**: Current config without gradient accumulation
2. **Static accumulation**: Always accumulate with 2 steps
3. **Dynamic accumulation**: Only accumulate when >250k muons
4. **Adaptive accumulation**: Adjust steps based on batch size

Metrics to track:
- Peak VRAM usage
- Training throughput (batches/sec)
- Time to OOM (if any)
- Model convergence (Wasserstein distance)
- Total training time to fixed batch count

## Expected Outcomes

With 2x gradient accumulation on large batches:
- **Peak memory reduction**: ~30-40% for affected batches
- **Throughput impact**: ~15-25% slower overall (only large batches affected)
- **OOM prevention**: Should eliminate crashes at 3700-3800 batch mark
- **Training quality**: No degradation expected (same effective batch size)

## References

- [PyTorch Gradient Accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)
- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028) - Improved Training of Wasserstein GANs
- [Memory-Efficient Training](https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-accumulation)

## Implementation Checklist

- [ ] Add command-line arguments for accumulation control
- [ ] Implement batch splitting logic (preserve event boundaries)
- [ ] Modify critic training loop for accumulation
- [ ] Modify generator training loop for accumulation
- [ ] Update gradient penalty calculation
- [ ] Add memory/accumulation logging
- [ ] Test on small dataset for correctness
- [ ] Benchmark memory savings
- [ ] Validate convergence matches baseline
- [ ] Document performance characteristics
- [ ] Update training documentation

## Next Steps

1. **Quick win**: Implement dynamic accumulation (option 3) - only accumulates for large batches
2. **Testing**: Run 5000 batch test with accumulation vs without
3. **Tune threshold**: Adjust `--accumulation-muon-threshold` based on results
4. **Monitor**: Watch for any training instability or convergence issues

If gradient accumulation proves effective, consider extending to other scenarios:
- Multi-GPU training with gradient accumulation across GPUs
- Mixed precision training (FP16) combined with accumulation
- Larger models with reduced batch sizes
