import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np


# ==========================================
# 2. MULTIPLICITY PREDICTOR
# ==========================================
class MultiplicityPredictor(nn.Module):
    """Predicts event multiplicity from event-level conditions.
    
    Args:
        cond_dim: Dimension of event conditions (default: 4)
        
    Returns in forward():
        Tensor of shape [batch_size, 1] containing log10(multiplicity) predictions
    """
    def __init__(self, cond_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 64), nn.ELU(),
            nn.Linear(64, 128),      nn.ELU(),
            nn.Linear(128, 64),      nn.ELU(),
            nn.Linear(64, 1)         # Output: log10(N)
        )
    
    def forward(self, cond):
        """Predict log10 of muon multiplicity.
        
        Args:
            cond: Event conditions [batch_size, cond_dim]
            
        Returns:
            Predicted log10(multiplicity) [batch_size, 1]
        """
        return self.net(cond)

# ==========================================
# 3. SCALABLE GENERATOR (Flat Output)
# ==========================================
class ScalableGenerator(nn.Module):
    """Generative model that produces muons in a flattened representation.
    
    Uses a two-stage architecture:
    1. Global network: Processes event-level context from conditions + noise
    2. Local network: Generates individual muon features from global context + local noise
    
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
        
        # Global Context Network: Aggregates event-level information
        # Input: [global_noise, conditions, normalized_multiplicity]
        self.global_net = nn.Sequential(
            nn.Linear(self.latent_dim_global + self.cond_dim + 1, 128),  # +1 for scaled multiplicity
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

    def forward(self, conditions, N_target_list):
        """Generate muons for a batch of events.
        
        Args:
            conditions: Event conditions [batch_size, cond_dim]
            N_target_list: List of muon counts per event [N1, N2, ..., N_batch]
            
        Returns:
            flat_muons: All generated muons [total_muons, feature_dim]
            batch_index: Batch assignment for each muon [total_muons]
        """
        batch_size = conditions.size(0)

        # Accept either a Python list of ints or a tensor of counts.
        if isinstance(N_target_list, torch.Tensor):
            counts_list = [int(x) for x in N_target_list.detach().tolist()]
        else:
            counts_list = [int(x) for x in N_target_list]

        if any(n < 0 for n in counts_list):
            raise ValueError("Muon counts must be non-negative")

        total_muons = int(sum(counts_list))
        
        # Validate inputs
        assert batch_size == len(counts_list), "Batch size mismatch with N_target_list"

        # If the whole batch has zero muons, return empty tensors.
        if total_muons == 0:
            empty_muons = torch.empty((0, self.feature_dim), device=self.device)
            empty_idx = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty_muons, empty_idx
        
        # 1. Generate Global Context per Event
        global_noise = torch.randn(batch_size, self.latent_dim_global).to(self.device)
        # Normalize multiplicity using log10 to match other log-transformed features (energy, area)
        # +1 avoids log(0) for edge cases
        # To convert back: round(10^network_output) gives integer multiplicity
        multiplicity_normalized = torch.log10(
            torch.tensor(counts_list, device=self.device).unsqueeze(1).float() + 1
        )
        
        global_input = torch.cat([global_noise, conditions, multiplicity_normalized], dim=1)
        event_context = self.global_net(global_input)  # [batch_size, HIDDEN_DIM]
        
        # 2. Expand Event Context to Match Individual Muons
        # Example: If batch has [2, 3] muons, repeat event 0 twice and event 1 three times
        repeats = torch.tensor(counts_list, device=self.device)
        flat_context = torch.repeat_interleave(event_context, repeats, dim=0)  # [total_muons, HIDDEN_DIM]
        
        # 3. Generate Independent Noise for Each Muon
        local_noise = torch.randn(total_muons, self.latent_dim_local).to(self.device)
        
        # 4. Generate Muon Features
        local_input = torch.cat([flat_context, local_noise], dim=1)
        flat_muons = self.local_net(local_input)  # [total_muons, feature_dim]
        
        # 5. Create Batch Index for Scatter Operations
        # Maps each muon back to its event: e.g., [0, 0, 1, 1, 1] for [2, 3] muons per event
        batch_index = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device), repeats
        )
        
        return flat_muons, batch_index

# ==========================================
# 4. SCALABLE CRITIC (Scatter Reduce)
# ==========================================
class ScalableCritic(nn.Module):
    """Discriminator that evaluates realism of event-level muon distributions.
    
    Architecture:
    1. Point network: Evaluates each muon independently with its event context
    2. Max pooling: Aggregates muon scores within each event (captures outliers)
    3. Decision network: Final event-level classification
    """
    def __init__(self, feat_dim=3, cond_dim=4, device="cpu"):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.cond_dim = cond_dim
        self.device = device

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
        
        # 3. Aggregate Muon Scores via Max Pooling
        # Max pooling captures outliers and unusual muons that violate physics
        # (Requires PyTorch 1.12+; see torch_scatter.scatter_max for older versions)
        global_feats = torch.full((batch_size, 128), -1e9, device=self.device)
        global_feats.scatter_reduce_(
            0, batch_index.unsqueeze(1).expand(-1, 128), point_feats, reduce="amax"
        )
        
        # 4. Final Event-Level Scoring
        decision_input = torch.cat([global_feats, conditions], dim=1)
        score = self.decision_net(decision_input)
        
        return score

# ==========================================
# 5. SCALABLE GRADIENT PENALTY
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
        Scalar gradient penalty loss
    """
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
    gradient_penalty = ((grad_norms - 1) ** 2).mean()
    
    return gradient_penalty

# ==========================================
# 6. TRAINING LOOP DEMO
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
    N_list,
    lambda_gp=10,
    critic_steps: int = 1,
    device="cpu",
):
    """Single training step for Wasserstein GAN with gradient penalty.
    
    Args:
        gen: Generator model
        crit: Critic model
        opt_G: Generator optimizer
        opt_C: Critic optimizer
        real_muons_flat: Real muon samples [total_muons, feature_dim]
        real_batch_idx: Batch index for real muons [total_muons]
        real_cond: Event conditions [batch_size, cond_dim]
        N_list: Muon counts per event [batch_size]
        lambda_gp: Gradient penalty weight
        device: Device to run computations on
        
    Returns:
        Tuple of (critic_loss, generator_loss) as scalars
    """
    batch_size = real_cond.size(0)
    
    critic_steps = int(max(1, critic_steps))
    loss_critic = None

    # ===== Update Critic =====
    for _ in range(critic_steps):
        opt_C.zero_grad()

        # Generate fake samples
        fake_muons_flat, fake_batch_idx = gen(real_cond, N_list)

        # Score real and fake samples (detach fakes to prevent generator gradient flow)
        real_score = crit(real_muons_flat, real_batch_idx, real_cond, batch_size)
        fake_score = crit(fake_muons_flat.detach(), fake_batch_idx, real_cond, batch_size)

        # Compute Wasserstein loss and gradient penalty
        gradient_penalty = compute_gp_flat(
            crit,
            real_muons_flat,
            fake_muons_flat.detach(),
            real_batch_idx,
            real_cond,
            batch_size,
            device=device,
        )

        # Wasserstein distance: minimize (fake - real)
        loss_critic = fake_score.mean() - real_score.mean() + lambda_gp * gradient_penalty
        loss_critic.backward()
        opt_C.step()
    
    # ===== Update Generator =====
    opt_G.zero_grad()
    
    # Re-evaluate fakes with gradients enabled for generator
    fake_score_G = crit(fake_muons_flat, fake_batch_idx, real_cond, batch_size)
    loss_generator = -fake_score_G.mean()  # Maximize critic score
    
    loss_generator.backward()
    opt_G.step()
    
    return float(loss_critic.item()) if loss_critic is not None else 0.0, float(loss_generator.item())

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
