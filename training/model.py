import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np


# ==========================================
# 1. SCALABLE GENERATOR (Flat Output)
# ==========================================
class ScalableGenerator(nn.Module):
    """Generative model that produces muons in a flattened representation.
    
    Uses a three-stage architecture:
    1. Multiplicity network: Predicts muon counts from event conditions
    2. Global network: Processes event-level context from conditions + noise + predicted multiplicity
    3. Local network: Generates individual muon features from global context + local noise
    
    The "flat" representation concatenates all muons from a batch into a single tensor
    to enable efficient processing with PyTorch's scatter operations.
    """
    def __init__(self, cond_dim=4, feat_dim=3, 
                 latent_dim_global=32, latent_dim_local=16, 
                 hidden_dim=256,
                 device="cpu"):
        super().__init__()

        self.cond_dim = cond_dim
        self.feature_dim = feat_dim
        self.latent_dim_global = latent_dim_global
        self.latent_dim_local = latent_dim_local
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Multiplicity Predictor: Predicts log10(N) from conditions
        self.multiplicity_net = nn.Sequential(
            nn.Linear(self.cond_dim, 64), nn.ELU(),
            nn.Linear(64, 128),           nn.ELU(),
            nn.Linear(128, 64),           nn.ELU(),
            nn.Linear(64, 1)              # Output: log10(N)
        )
        
        # Global Context Network: Aggregates event-level information
        # Input: [global_noise, conditions, predicted_multiplicity]
        self.global_net = nn.Sequential(
            nn.Linear(self.latent_dim_global + self.cond_dim + 1, 128),  # +1 for predicted multiplicity
            nn.LayerNorm(128), nn.LeakyReLU(0.2),
            nn.Linear(128, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.LeakyReLU(0.2),
        )
        
        # Local Generator Network: Produces per-muon features
        # Input: [global_context, local_noise]
        self.local_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim_local, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.feature_dim)
        )

    def forward(self, conditions):
        """Generate muons for a batch of events.
        
        Args:
            conditions: Event conditions [batch_size, cond_dim]
            
        Returns:
            flat_muons: All generated muons [total_muons, feature_dim]
            batch_index: Batch assignment for each muon [total_muons]
        """
        batch_size = conditions.size(0)

        # 1. Predict multiplicity from conditions
        multiplicity_log10 = self.multiplicity_net(conditions)  # [batch_size, 1]
        # Sanitize to avoid NaNs/Infs propagating
        mul_log10_clean = torch.nan_to_num(multiplicity_log10, nan=0.0, posinf=0.0, neginf=0.0)
        # Convert log10(N+1) -> counts and keep on device (no CPU sync)
        counts = torch.pow(10.0, mul_log10_clean.squeeze(1)) - 1
        counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        counts = counts.clamp_min(0).round().long()  # [batch_size]

        # No host sync here; handle zero-muon batches naturally via empty tensors downstream.
        
        # 2. Generate Global Context per Event
        global_noise = torch.randn(batch_size, self.latent_dim_global, device=self.device)
        # Use the predicted multiplicity (log10 normalized) in the global context
        multiplicity_normalized = mul_log10_clean
        
        global_input = torch.cat([global_noise, conditions, multiplicity_normalized], dim=1)
        event_context = self.global_net(global_input)  # [batch_size, HIDDEN_DIM]
        
        # 3. Expand Event Context to Match Individual Muons
        # Example: If batch has [2, 3] muons, repeat event 0 twice and event 1 three times
        repeats = counts  # [batch_size]
        flat_context = torch.repeat_interleave(event_context, repeats, dim=0)  # [total_muons, HIDDEN_DIM]
        
        # 4. Generate Independent Noise for Each Muon
        # Generate per-muon noise directly on device without host sync
        local_noise = torch.empty((flat_context.size(0), self.latent_dim_local), device=self.device).normal_()
        
        # 5. Generate Muon Features
        local_input = torch.cat([flat_context, local_noise], dim=1)
        flat_muons = self.local_net(local_input)  # [total_muons, feature_dim]
        
        # 6. Create Batch Index for Scatter Operations
        # Maps each muon back to its event: e.g., [0, 0, 1, 1, 1] for [2, 3] muons per event
        batch_index = torch.repeat_interleave(torch.arange(batch_size, device=self.device), repeats)
        
        return flat_muons, batch_index

# ==========================================
# 2. SCALABLE CRITIC (Scatter Reduce)
# ==========================================
class ScalableCritic(nn.Module):
    """Discriminator that evaluates realism of event-level muon distributions.
    
    Architecture:
    1. Point network: Evaluates each muon independently with its event context
    2. Max pooling: Aggregates muon scores within each event (captures outliers)
    3. Decision network: Final event-level classification
    """
    def __init__(self, feat_dim=3, cond_dim=4, device="cpu", pooling_mode: str = "amax"):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.cond_dim = cond_dim
        self.device = device
        self.pooling_mode = str(pooling_mode or "amax").lower()

        # Per-Muon Feature Processor: Conditioned on event context
        self.point_net = nn.Sequential(
            nn.Linear(feat_dim + cond_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2)
        )
        
        # Event-Level Decision Network: Operates on aggregated muon features
        self.decision_net = nn.Sequential(
            nn.Linear(128 + cond_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, flat_muons, batch_index, conditions, batch_size):
        """Score events based on muon distributions.
        
        Args:
            flat_muons: Concatenated muons from all events [total_muons, feature_dim]
            batch_index: Event assignment for each muon [total_muons]
            conditions: Event conditions [batch_size, cond_dim]
            batch_size: Number of events in batch
            
        Returns:
            Event-level scores [batch_size, 1] (higher = more real)
        """
        # 1. Broadcast Event Conditions to Muon Level
        # Each muon gets paired with its event's conditions
        flat_cond = conditions[batch_index]  # [total_muons, cond_dim]
        
        # 2. Evaluate Each Muon in Event Context
        point_input = torch.cat([flat_muons, flat_cond], dim=1)
        point_feats = self.point_net(point_input)  # [total_muons, 128]
        # Sanitize to prevent NaNs/Infs from propagating through amax pooling
        point_feats = torch.nan_to_num(point_feats, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Aggregate Muon Scores via Pooling
        if self.pooling_mode == "amax":
            global_feats = torch.full((batch_size, 128), -1e9, device=self.device)
            global_feats.scatter_reduce_(
                0, batch_index.unsqueeze(1).expand(-1, 128), point_feats, reduce="amax"
            )
        elif self.pooling_mode == "mean":
            global_feats = torch.zeros((batch_size, 128), device=self.device)
            global_feats.scatter_reduce_(
                0, batch_index.unsqueeze(1).expand(-1, 128), point_feats, reduce="mean"
            )
        else:
            # Fallback to amax if an unknown mode is provided
            global_feats = torch.full((batch_size, 128), -1e9, device=self.device)
            global_feats.scatter_reduce_(
                0, batch_index.unsqueeze(1).expand(-1, 128), point_feats, reduce="amax"
            )
        # Ensure finite values post pooling (events without muons remain at -1e9)
        global_feats = torch.nan_to_num(global_feats, nan=-1e9, posinf=1e9, neginf=-1e9)
        
        # 4. Final Event-Level Scoring
        decision_input = torch.cat([global_feats, conditions], dim=1)
        score = self.decision_net(decision_input)
        
        # Normalize score by batch size to keep Wasserstein loss in reasonable range
        # This prevents gradient explosion when batch_size is large
        score = score / max(1.0, float(batch_size) ** 0.5)
        
        # Clamp to prevent unbounded growth of critic weights
        # Wasserstein distance should be in [-1000, 1000] range for stable training
        score = torch.clamp(score, min=-1000.0, max=1000.0)
        
        return score

# ==========================================
# 3. SCALABLE GRADIENT PENALTY
# ==========================================
def compute_gp_flat(critic, real_flat, fake_flat, batch_index, conditions, batch_size, device="cpu"):
    """Compute Wasserstein gradient penalty for discriminator.
    
    Enforces 1-Lipschitz constraint on critic by penalizing gradients on
    interpolated samples. Uses per-muon gradient norms which is more stable
    for PointNet-style architectures than event-level aggregation.
    
    Args:
        critic: Discriminator network
        real_flat: Real muons [total_muons, feature_dim]
        fake_flat: Generated muons [total_muons, feature_dim]
        batch_index: Event assignment [total_muons]
        conditions: Event conditions [batch_size, cond_dim]
        batch_size: Number of events
        device: Device to run computations on
        
    Returns:
        Scalar gradient penalty loss (0 if no fake samples)
    """
    # Handle case where generator produces no samples (early in training)
    if fake_flat.numel() == 0:
        return torch.tensor(0.0, device=device)
    
    # Sample random interpolation weights per event
    alpha = torch.rand(batch_size, 1, device=device)
    alpha_expanded = alpha[batch_index]  # Broadcast to muon level
    
    # Linear interpolation between real and fake samples
    interpolates = (alpha_expanded * real_flat + (1 - alpha_expanded) * fake_flat).requires_grad_(True)
    
    # Score interpolated samples
    d_interpolates = critic(interpolates, batch_index, conditions, batch_size)
    
    # Compute gradients
    fake_labels = torch.ones(batch_size, 1, device=device)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_labels,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Penalize per-muon gradient norms (more stable than event aggregation)
    # This encourages locally linear critic around data manifold
    grad_norms = gradients.norm(2, dim=1)  # [total_muons]
    # Replace non-finite gradient norms and softly clamp to curb explosions
    grad_norms = torch.nan_to_num(grad_norms, nan=0.0, posinf=10.0, neginf=0.0).clamp_max(10.0)
    gradient_penalty = ((grad_norms - 1) ** 2).mean()
    
    return gradient_penalty

# ==========================================
# 4. TRAINING LOOP
# ==========================================
# Initialize models and optimizers
# Moved to if __name__ == "__main__": block at the end of file to avoid global variables


def train_step_scalable(
    gen,
    crit,
    opt_G,
    opt_C,
    real_muons_flat,
    real_batch_idx,
    real_cond,
    real_counts,
    lambda_gp=10,
    critic_steps: int = 1,
    gp_max_pairs: int = 0,
    gp_sample_fraction: float = 0.0,
    gp_every: int = 1,
    grad_clip_norm: float = 0.0,
    device="cpu",
):
    """Single training step for Wasserstein GAN with gradient penalty.
    
    Trains three networks:
    1. Multiplicity predictor (gen.multiplicity_net): to predict true counts from conditions
    2. Critic: to distinguish real from generated muons
    3. Generator: to fool the critic
    
    Args:
        gen: Generator model (includes multiplicity_net)
        crit: Critic model
        opt_G: Generator optimizer
        opt_C: Critic optimizer
        real_muons_flat: Real muon samples [total_muons, feature_dim]
        real_batch_idx: Batch index for real muons [total_muons]
        real_cond: Event conditions [batch_size, cond_dim]
        real_counts: True muon counts per event [batch_size]
        lambda_gp: Gradient penalty weight
        critic_steps: Number of critic updates per generator update
        device: Device to run computations on
        
    Returns:
        Tuple of (critic_loss, generator_loss, multiplicity_loss, w_gap) as scalars
    """
    batch_size = real_cond.size(0)
    
    critic_steps = int(max(1, critic_steps))
    loss_critic = None
    loss_multiplicity = None
    w_gap_val = 0.0

    # ===== Train Multiplicity Predictor =====
    # Target: log10(real_counts + 1) to match network's output
    if isinstance(real_counts, torch.Tensor):
        real_counts_tensor = real_counts.float().to(device).unsqueeze(1) if real_counts.dim() == 1 else real_counts.to(device)
    else:
        real_counts_tensor = torch.tensor(real_counts, device=device, dtype=torch.float32).unsqueeze(1)
    target_multiplicity = torch.log10(real_counts_tensor + 1)
    
    opt_G.zero_grad()
    pred_multiplicity = gen.multiplicity_net(real_cond)
    loss_multiplicity = nn.functional.mse_loss(pred_multiplicity, target_multiplicity)
    loss_multiplicity.backward()
    if float(grad_clip_norm) > 0.0:
        torch.nn.utils.clip_grad_norm_(gen.parameters(), float(grad_clip_norm))
    opt_G.step()

    # ===== Update Critic =====
    for ci in range(int(critic_steps)):
        opt_C.zero_grad()

        # Generate fake samples (generator predicts its own multiplicity)
        fake_muons_flat, fake_batch_idx = gen(real_cond)

        # Score real and fake samples (detach fakes to prevent generator gradient flow)
        real_score = crit(real_muons_flat, real_batch_idx, real_cond, batch_size)
        fake_score = crit(fake_muons_flat.detach(), fake_batch_idx, real_cond, batch_size)
        try:
            w_gap_val = float((real_score.mean() - fake_score.mean()).item())
        except Exception:
            w_gap_val = 0.0

        # Build per-event aligned subsets for gradient penalty efficiently
        # Pair up min(count_real, count_fake) muons per event using sorted indices
        apply_gp = (lambda_gp > 0) and (int(gp_every) > 0) and ((ci % int(gp_every)) == 0)
        with torch.no_grad():
            if apply_gp:
                # Counts per event
                real_counts_ev = torch.bincount(real_batch_idx, minlength=batch_size)
                fake_counts_ev = torch.bincount(fake_batch_idx, minlength=batch_size)
                k_ev = torch.minimum(real_counts_ev, fake_counts_ev)

                # Sort indices by event once (avoids repeated nonzero scans)
                real_sorted_order = torch.argsort(real_batch_idx)
                fake_sorted_order = torch.argsort(fake_batch_idx)
                real_sorted_idx = real_sorted_order
                fake_sorted_idx = fake_sorted_order

                # Offsets per event into the sorted index arrays
                real_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
                fake_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
                if batch_size > 1:
                    real_offsets[1:] = torch.cumsum(real_counts_ev[:-1], dim=0)
                    fake_offsets[1:] = torch.cumsum(fake_counts_ev[:-1], dim=0)

                # Collect aligned slices
                idx_real_list = []
                idx_fake_list = []
                batch_sub_list = []
                nonzero_events = (k_ev > 0).nonzero(as_tuple=False).squeeze(1)
                # Batch transfer to CPU before loop to avoid repeated GPU syncs
                k_ev_cpu = k_ev.cpu()
                real_offsets_cpu = real_offsets.cpu()
                fake_offsets_cpu = fake_offsets.cpu()
                for e in nonzero_events.tolist():
                    k = int(k_ev_cpu[e].item())
                    r_start = int(real_offsets_cpu[e].item())
                    f_start = int(fake_offsets_cpu[e].item())
                    idx_real_list.append(real_sorted_idx[r_start:r_start + k])
                    idx_fake_list.append(fake_sorted_idx[f_start:f_start + k])
                    batch_sub_list.append(torch.full((k,), e, dtype=torch.long, device=device))

                if idx_real_list:
                    idx_r = torch.cat(idx_real_list)
                    idx_f = torch.cat(idx_fake_list)
                    batch_idx_sub = torch.cat(batch_sub_list)
                    # Optional subsampling to limit GP cost
                    n_pairs = int(idx_r.numel())
                    target_n = n_pairs
                    if int(gp_max_pairs) > 0:
                        target_n = min(target_n, int(gp_max_pairs))
                    if float(gp_sample_fraction) > 0.0 and float(gp_sample_fraction) < 1.0:
                        frac_n = max(1, int(n_pairs * float(gp_sample_fraction)))
                        target_n = min(target_n, frac_n)
                    if target_n < n_pairs:
                        perm = torch.randperm(n_pairs, device=device)[:target_n]
                        idx_r = idx_r[perm]
                        idx_f = idx_f[perm]
                        batch_idx_sub = batch_idx_sub[perm]

                    real_sub = real_muons_flat[idx_r]
                    fake_sub = fake_muons_flat.detach()[idx_f]
                else:
                    # No pairs available; skip penalty
                    real_sub = torch.empty((0, real_muons_flat.shape[1]), device=device)
                    fake_sub = torch.empty((0, real_muons_flat.shape[1]), device=device)
                    batch_idx_sub = torch.empty((0,), dtype=torch.long, device=device)
            else:
                # GP not applied on this critic step
                real_sub = torch.empty((0, real_muons_flat.shape[1]), device=device)
                fake_sub = torch.empty((0, real_muons_flat.shape[1]), device=device)
                batch_idx_sub = torch.empty((0,), dtype=torch.long, device=device)

        # Compute Wasserstein loss and gradient penalty on aligned subsets
        gradient_penalty = compute_gp_flat(
            crit,
            real_sub,
            fake_sub,
            batch_idx_sub,
            real_cond,
            batch_size,
            device=device,
        ) if apply_gp else torch.tensor(0.0, device=device)

        # Wasserstein distance: minimize (fake - real)
        loss_critic = fake_score.mean() - real_score.mean() + lambda_gp * gradient_penalty
        loss_critic.backward()
        if float(grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(crit.parameters(), float(grad_clip_norm))
        opt_C.step()
    
    # ===== Update Generator =====
    opt_G.zero_grad()
    
    # Re-evaluate fakes with gradients enabled for generator
    fake_score_G = crit(fake_muons_flat, fake_batch_idx, real_cond, batch_size)
    loss_generator = -fake_score_G.mean()  # Maximize critic score
    
    loss_generator.backward()
    if float(grad_clip_norm) > 0.0:
        torch.nn.utils.clip_grad_norm_(gen.parameters(), float(grad_clip_norm))
    opt_G.step()

    return (
        float(loss_critic.item()) if loss_critic is not None else 0.0,
        float(loss_generator.item()),
        float(loss_multiplicity.item()) if loss_multiplicity is not None else 0.0,
        float(w_gap_val),
    )

if __name__ == "__main__":
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_LATENT_DIM = 32
    LOCAL_LATENT_DIM  = 16
    COND_DIM          = 4
    FEAT_DIM          = 3
    HIDDEN_DIM        = 256
    LAMBDA_GP         = 10

    # Initialize models and optimizers
    gen = ScalableGenerator(cond_dim=COND_DIM, feat_dim=FEAT_DIM, 
                            latent_dim_global=GLOBAL_LATENT_DIM, latent_dim_local=LOCAL_LATENT_DIM, 
                            hidden_dim=HIDDEN_DIM, device=DEVICE).to(DEVICE)
    crit = ScalableCritic(feat_dim=FEAT_DIM, cond_dim=COND_DIM, device=DEVICE).to(DEVICE)

    # Use Adam with momentum=0 (standard for Wasserstein GANs)
    opt_G = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_C = optim.Adam(crit.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    print("Models initialized successfully.")
