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
                 device=None): # device arg kept for back-compat but ignored ideally
        super().__init__()

        self.cond_dim = cond_dim
        self.feature_dim = feat_dim
        self.latent_dim_global = latent_dim_global
        self.latent_dim_local = latent_dim_local
        self.hidden_dim = hidden_dim
        
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
        
        # Initialize weights for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Conservative gain for stability
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, conditions):
        """Generate muons for a batch of events.
        
        Args:
            conditions: Event conditions [batch_size, cond_dim]
            
        Returns:
            flat_muons: All generated muons [total_muons, feature_dim]
            batch_index: Batch assignment for each muon [total_muons]
        """
        batch_size = conditions.size(0)
        device = conditions.device

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
        global_noise = torch.randn(batch_size, self.latent_dim_global, device=device)
        # Use the predicted multiplicity (log10 normalized) in the global context
        multiplicity_normalized = mul_log10_clean
        
        # Concatenate inputs for global network
        global_input = torch.cat([global_noise, conditions, multiplicity_normalized], dim=1)
        
        event_context = self.global_net(global_input)  # [batch_size, HIDDEN_DIM]
        
        # 3. Expand Event Context to Match Individual Muons
        # Example: If batch has [2, 3] muons, repeat event 0 twice and event 1 three times
        repeats = counts  # [batch_size]
        flat_context = torch.repeat_interleave(event_context, repeats, dim=0)  # [total_muons, HIDDEN_DIM]
        
        # 4. Generate Independent Noise for Each Muon
        # Generate per-muon noise directly on device without host sync
        local_noise = torch.empty((flat_context.size(0), self.latent_dim_local), device=device).normal_()
        
        # 5. Generate Muon Features
        # Concatenate global context and local noise
        local_input = torch.cat([flat_context, local_noise], dim=1)
        
        flat_muons = self.local_net(local_input)  # [total_muons, feature_dim]
        
        # 6. Create Batch Index for Scatter Operations
        # Maps each muon back to its event: e.g., [0, 0, 1, 1, 1] for [2, 3] muons per event
        batch_index = torch.repeat_interleave(torch.arange(batch_size, device=device), repeats)
        
        return flat_muons, batch_index

    def generate_with_counts(self, conditions, counts):
        """Generate muons with externally provided per-event counts.
        
        Args:
            conditions: [batch_size, cond_dim] event conditions
            counts: [batch_size] desired muon counts per event (int/long)
        
        Returns:
            flat_muons: [total_muons, feature_dim]
            batch_index: [total_muons]
        """
        batch_size = conditions.size(0)
        device = conditions.device

        # Sanitize counts
        if not isinstance(counts, torch.Tensor):
            counts = torch.tensor(counts, device=device, dtype=torch.long)
        else:
            counts = counts.to(device)
        counts = torch.nan_to_num(counts.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0).round().long()

        # Global context uses log10(N+1) to match training semantics
        multiplicity_normalized = torch.log10(counts.float().unsqueeze(1) + 1.0)

        # Global Context per Event
        global_noise = torch.randn(batch_size, self.latent_dim_global, device=device)
        global_input = torch.cat([global_noise, conditions, multiplicity_normalized], dim=1)
        event_context = self.global_net(global_input)  # [batch_size, HIDDEN_DIM]

        # Expand Event Context for each muon
        repeats = counts  # [batch_size]
        # Avoid .item() sync - check if any events have muons without CPU transfer
        if repeats.sum() == 0:
            # Return empty tensors on device
            empty_mu = torch.empty((0, self.feature_dim), device=device)
            empty_idx = torch.empty((0,), dtype=torch.long, device=device)
            return empty_mu, empty_idx

        flat_context = torch.repeat_interleave(event_context, repeats, dim=0)

        # Per-muon noise and local generation
        local_noise = torch.empty((flat_context.size(0), self.latent_dim_local), device=device).normal_()
        local_input = torch.cat([flat_context, local_noise], dim=1)
        flat_muons = self.local_net(local_input)

        # Batch index mapping
        batch_index = torch.repeat_interleave(torch.arange(batch_size, device=device), repeats)

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
    def __init__(self, feat_dim=3, cond_dim=4, device=None, pooling_mode: str = "amax"):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.cond_dim = cond_dim
        self.pooling_mode = str(pooling_mode or "amax").lower()
        
        # Pre-allocate pooling buffer (will be resized as needed, but reduces allocations)
        self._pooling_buffer = None
        self._pooling_buffer_size = 0
        
        # Running statistics for score normalization (decoupled from batch size)
        self.register_buffer("_score_norm_ema", torch.tensor(1.0)) 
        self._score_norm_momentum = 0.99  # How much to weight old statistics

        # Per-Muon Feature Processor: Conditioned on event context
        self.point_net = nn.Sequential(
            nn.Linear(feat_dim + cond_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2)
        )
        
        # Event-Level Decision Network: Operates on aggregated muon features
        self.decision_net = nn.Sequential(
            nn.Linear(128 + cond_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  # Normalize before final layer to keep values bounded
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        
        # Initialize weights for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Conservative gain for stability
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        device = conditions.device
        
        # 1. Broadcast Event Conditions to Muon Level
        # Each muon gets paired with its event's conditions
        flat_cond = conditions[batch_index]  # [total_muons, cond_dim]
        
        # 2. Evaluate Each Muon in Event Context
        point_input = torch.cat([flat_muons, flat_cond], dim=1)
        point_feats = self.point_net(point_input)  # [total_muons, 128]
        # Sanitize to prevent NaNs/Infs from propagating through amax pooling
        point_feats = torch.nan_to_num(point_feats, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Aggregate Muon Scores via Pooling
        # Reuse pre-allocated buffer to reduce aten::full calls (major bottleneck)
        required_size = batch_size * 128
        
        # Update buffer device if needed
        if self._pooling_buffer is not None and self._pooling_buffer.device != device:
             self._pooling_buffer = None
             self._pooling_buffer_size = 0

        # Resize buffer if needed (rare, only when batch_size increases)
        if self._pooling_buffer is None or self._pooling_buffer_size < required_size:
            # Allocate with 20% headroom to reduce future reallocations
            alloc_size = int(required_size * 1.2)
            if self.pooling_mode == "amax":
                self._pooling_buffer = torch.full((alloc_size,), -1e9, device=device)
            else:
                self._pooling_buffer = torch.zeros((alloc_size,), device=device)
            self._pooling_buffer_size = alloc_size
        
        # Clone first to avoid inplace modification issues with autograd
        # (slight overhead but still much faster than torch.full each time)
        global_feats = self._pooling_buffer[:required_size].clone().view(batch_size, 128)
        
        # Now safe to use inplace operations on the clone
        if self.pooling_mode == "amax":
            global_feats.fill_(-1e9)
            global_feats.scatter_reduce_(
                0, batch_index.unsqueeze(1).expand(-1, 128), point_feats, reduce="amax"
            )
        elif self.pooling_mode == "mean":
            global_feats.fill_(0.0)
            global_feats.scatter_reduce_(
                0, batch_index.unsqueeze(1).expand(-1, 128), point_feats, reduce="mean"
            )
        else:
            # Fallback to amax if an unknown mode is provided
            global_feats.fill_(-1e9)
            global_feats.scatter_reduce_(
                0, batch_index.unsqueeze(1).expand(-1, 128), point_feats, reduce="amax"
            )
        # Ensure finite values post pooling (events without muons remain at -1e9)
        global_feats = torch.nan_to_num(global_feats, nan=-1e9, posinf=1e9, neginf=-1e9)
        
        # 4. Final Event-Level Scoring
        decision_input = torch.cat([global_feats, conditions], dim=1)
        score = self.decision_net(decision_input)
        
        # Update running norm statistic (momentum = 0.99 means slow decay)
        # This keeps scores normalized without depending on batch size
        score_magnitude = score.abs().mean().detach()
        # Ensure ema is on correct device from init or register_buffer
        # self._score_norm_ema should be managed by module via register_buffer
        
        with torch.no_grad():
             self._score_norm_ema.mul_(self._score_norm_momentum).add_(score_magnitude, alpha=1 - self._score_norm_momentum)
        
        # Normalize score by running statistics, then soft-saturate
        # norm_eps prevents division by near-zero early in training
        norm_eps = 0.1
        normalized_score = score / max(float(self._score_norm_ema.item()), norm_eps)
        
        # Clamp + tanh ensures stability: prevents saturation but keeps bounded
        score = torch.clamp(normalized_score, -5.0, 5.0)
        score = 10.0 * torch.tanh(score / 5.0)  # Map [-5, 5] → [-10, 10]
        
        return score

# ==========================================
# 3. SCALABLE GRADIENT PENALTY
# ==========================================
def compute_gp_flat(critic, real_flat, fake_flat, batch_index, conditions, batch_size):
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
        
    Returns:
        Scalar gradient penalty loss (0 if no fake samples)
    """
    device = conditions.device
    
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
        create_graph=True, #gemini claims this is required # No second-order derivatives; preserves gradient flow & saves memory
        retain_graph=True,  # Safe to free after grad computation
        only_inputs=True
    )[0]
    
    # Penalize per-muon gradient norms (more stable than event aggregation)
    # This encourages locally linear critic around data manifold
    grad_norms = gradients.norm(2, dim=1)  # [total_muons]
    # Replace non-finite gradient norms and softly clamp to curb explosions
    grad_norms = torch.nan_to_num(grad_norms, nan=0.0, posinf=10.0, neginf=0.0).clamp_max(10.0)
    gradient_penalty = ((grad_norms - 1) ** 2).mean()
    
    # NO detach: gradient_penalty must flow back through critic to provide meaningful loss signal
    
    # Explicit cleanup: delete large intermediate tensors
    del interpolates, d_interpolates, gradients, grad_norms, fake_labels, alpha, alpha_expanded
    
    return gradient_penalty
