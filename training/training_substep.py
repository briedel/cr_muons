"""Helper function for running a single training substep.

This is extracted to avoid code duplication between the multi-file and sequential
training paths in train.py.
"""

import torch
from tqdm import tqdm
from model import train_step_scalable


def run_training_substep(
    sub_muons_feats: torch.Tensor,
    sub_batch_idx: torch.Tensor,
    sub_prims_feats: torch.Tensor,
    sub_counts: torch.Tensor,
    gen,
    crit,
    opt_G,
    opt_C,
    normalizer,
    device,
    non_blocking: bool,
    lambda_gp_cur: float,
    critic_steps_cur: int,
    gp_max_pairs: int,
    gp_sample_fraction: float,
    gp_every: int,
    grad_clip_norm: float,
    grad_accum_steps: int,
    profile_steps: int,
    steps_profiled: int,
    first_batch_printed: bool,
    batches_seen_global: int = 0,
):
    """Execute a single training substep.
    
    Args:
        sub_muons_feats: Muon features for this substep
        sub_batch_idx: Batch indices for muons
        sub_prims_feats: Primary particle features
        sub_counts: Count of muons per event
        gen: Generator model
        crit: Critic model
        opt_G: Generator optimizer
        opt_C: Critic optimizer
        normalizer: Data normalizer
        device: Torch device
        non_blocking: Whether to use non-blocking transfers
        lambda_gp_cur: Current gradient penalty weight
        critic_steps_cur: Current critic steps per generator step
        gp_max_pairs: Max pairs for gradient penalty
        gp_sample_fraction: Sampling fraction for gradient penalty
        gp_every: Apply GP every N critic steps
        grad_clip_norm: Gradient clipping norm
        grad_accum_steps: Gradient accumulation steps
        profile_steps: Number of steps to profile
        steps_profiled: Current count of profiled steps
        first_batch_printed: Whether first batch debug info was printed
        batches_seen_global: Global batch count for logging
        
    Returns:
        Tuple of (c_loss, g_loss, m_loss, w_gap, steps_profiled, first_batch_printed)
    """
    # Move to device
    sub_muons_feats = sub_muons_feats.to(device, non_blocking=non_blocking)
    sub_batch_idx = sub_batch_idx.to(device, non_blocking=non_blocking)
    sub_prims_feats = sub_prims_feats.to(device, non_blocking=non_blocking)
    sub_counts = sub_counts.to(device, non_blocking=non_blocking)

    # Normalize
    sub_muons_norm = normalizer.normalize_features(sub_muons_feats)
    sub_prims_norm = normalizer.normalize_primaries(sub_prims_feats)

    # Print first batch sample for debugging
    if (not first_batch_printed) and (sub_muons_feats.numel() > 0):
        first_batch_printed = True
        n_show = min(10, int(sub_muons_feats.shape[0]))
        tqdm.write("\n" + "="*80)
        tqdm.write("[First Training Batch Sample Inspector]")
        tqdm.write("="*80)
        tqdm.write(f"\n--- Muon Features (first {n_show} samples) ---")
        tqdm.write("Unnormalized:")
        for i in range(n_show):
            vals_str = "  ".join([f"{v:.6f}" for v in sub_muons_feats[i].cpu().numpy()])
            tqdm.write(f"  [{i}] {vals_str}")
        tqdm.write("\nNormalized:")
        for i in range(n_show):
            vals_str = "  ".join([f"{v:.6f}" for v in sub_muons_norm[i].detach().cpu().numpy()])
            tqdm.write(f"  [{i}] {vals_str}")
        n_prims_show = min(10, int(sub_prims_feats.shape[0]))
        tqdm.write(f"\n--- Primary Features (first {n_prims_show} events) ---")
        tqdm.write("Unnormalized:")
        for i in range(n_prims_show):
            vals_str = "  ".join([f"{v:.6f}" for v in sub_prims_feats[i].cpu().numpy()])
            tqdm.write(f"  [{i}] {vals_str}")
        tqdm.write("\nNormalized:")
        for i in range(n_prims_show):
            vals_str = "  ".join([f"{v:.6f}" for v in sub_prims_norm[i].detach().cpu().numpy()])
            tqdm.write(f"  [{i}] {vals_str}")
        tqdm.write("="*80 + "\n")

    # Optional profiling
    should_profile = (profile_steps > 0 and steps_profiled < profile_steps)
    prof = None
    if should_profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        )
        prof.__enter__()

    # Training step
    c_loss_local, g_loss_local, m_loss_local, w_gap_local, total_fake_muons_local = train_step_scalable(
        gen,
        crit,
        opt_G,
        opt_C,
        sub_muons_norm,
        sub_batch_idx,
        sub_prims_norm,
        sub_counts,
        lambda_gp=lambda_gp_cur,
        critic_steps=int(critic_steps_cur),
        gp_max_pairs=int(gp_max_pairs),
        gp_sample_fraction=float(gp_sample_fraction),
        gp_every=int(gp_every),
        grad_clip_norm=float(grad_clip_norm),
        grad_accum_steps=int(grad_accum_steps),
        device=device,
    )

    if should_profile and prof is not None:
        prof.__exit__(None, None, None)
        try:
            tqdm.write(f"[Profile step {steps_profiled+1}/{profile_steps}]")
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
            tqdm.write(table)
        except Exception as e:
            tqdm.write(f"[Profile error: {e}]")
        steps_profiled += 1

    # Optional debug logging
    if torch.cuda.is_available() and w_gap_local < 0:
        vram_used = torch.cuda.memory_allocated(device) / 1024**3
        vram_peak = torch.cuda.max_memory_reserved(device) / 1024**3
        tqdm.write(
            f"[Debug] batches={batches_seen_global} Substep losses: "
            f"c_loss={c_loss_local:.4f} g_loss={g_loss_local:.4f} "
            f"m_loss={m_loss_local:.4f} w_gap={w_gap_local:.4f}, "
            f"total_fake_muons={total_fake_muons_local}, "
            f"vram_used={vram_used:.4f} GB, vram_peak={vram_peak:.4f} GB"
        )

    return c_loss_local, g_loss_local, m_loss_local, w_gap_local, steps_profiled, first_batch_printed
