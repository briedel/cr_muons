import torch


def generate_event(gen, prims_feats: torch.Tensor, count: int, max_muons_per_event: int = 0):
    """Inference helper: generate muons for a single event.

    - Returns empty tensors if count <= 0.
    - Caps count to max_muons_per_event when > 0.

    Args:
        gen: ScalableGenerator (trained)
        prims_feats: [1, cond_dim] normalized primary features for one event
        count: desired muon count for the event
        max_muons_per_event: optional cap for safety; 0 disables

    Returns:
        flat_muons: [count, feature_dim] (or [0, feature_dim] if count<=0)
        batch_index: [count] (or empty)
    """
    device = getattr(gen, "device", "cpu")
    if max_muons_per_event and max_muons_per_event > 0:
        count = int(min(max(0, int(count)), int(max_muons_per_event)))
    else:
        count = int(max(0, int(count)))

    # Shape primaries to batch of 1 and keep on generator device
    prims_feats = prims_feats.to(device)
    if prims_feats.dim() == 1:
        prims_feats = prims_feats.unsqueeze(0)

    # Empty event -> return zero-sized tensors
    if count == 0:
        empty_mu = torch.empty((0, gen.feature_dim), device=device)
        empty_idx = torch.empty((0,), dtype=torch.long, device=device)
        return empty_mu, empty_idx

    counts = torch.tensor([count], device=device, dtype=torch.long)
    flat_muons, batch_index = gen.generate_with_counts(prims_feats, counts)
    return flat_muons, batch_index
