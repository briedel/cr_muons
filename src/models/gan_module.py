import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
from .components.gan_components import ScalableGenerator, ScalableCritic, compute_gp_flat
from .normalizer import GANNormalizer

class MuonGAN(pl.LightningModule):
    def __init__(self, 
                 cond_dim=4, 
                 feat_dim=3, 
                 latent_dim_global=32, 
                 latent_dim_local=16, 
                 hidden_dim=256, 
                 critic_pooling="amax",
                 lr=1e-4, 
                 beta1=0.0,
                 beta2=0.9,
                 critic_steps=5, 
                 lambda_gp=10.0,
                 grad_accum_steps=1,
                 grad_clip_norm=0.0,
                 gp_every=2,
                 gp_max_pairs=4096,
                 gp_sample_fraction=0.0,
                 max_muons_per_batch=0,
                 max_muons_per_event=0,
                 drop_empty_events=False,
                 outliers_dir=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.automatic_optimization = False
        
        # Physics normalization
        self.normalizer = GANNormalizer()
        
        # Histogram tracking
        self.last_real_feats = None
        self.last_fake_feats = None
        
        # Dynamic hyperparams (can be modified by callbacks)
        self.critic_steps = critic_steps
        self.lambda_gp = lambda_gp
        
        # Outlier writer
        self.outlier_writer = None
        if outliers_dir:
            from ..utils.data_utils import OutlierParquetWriter
            self.outlier_writer = OutlierParquetWriter(outliers_dir)
        
        self.generator = ScalableGenerator(
            cond_dim=cond_dim, 
            feat_dim=feat_dim, 
            latent_dim_global=latent_dim_global, 
            latent_dim_local=latent_dim_local, 
            hidden_dim=hidden_dim
        )
        
        # Init hack from train.py
        with torch.no_grad():
            if hasattr(self.generator.multiplicity_net, 'bias'):
                self.generator.multiplicity_net.bias.fill_(0.5)
            else:
                 # If it's a Sequential block, target the last layer [-1]
                 try:
                    self.generator.multiplicity_net[-1].bias.fill_(0.5)
                 except:
                    pass

        self.critic = ScalableCritic(
            feat_dim=feat_dim, 
            cond_dim=cond_dim, 
            pooling_mode=critic_pooling
        )

    def forward(self, conditions):
        return self.generator(conditions)

    def configure_optimizers(self):
        opt_c = optim.Adam(self.critic.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))
        opt_g = optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))
        return [opt_c, opt_g], []

    def training_step(self, batch, batch_idx):
        real_muons, real_batch_idx, prims, counts = batch
        
        # 1. Feature Slicing & Normalization (Data Prep)
        if prims.shape[1] == 6:
            conditions_raw = prims[:, 2:]
        else:
            conditions_raw = prims
            
        if real_muons.shape[1] == 5:
            real_muons_feats_raw = real_muons[:, 2:]
        else:
            real_muons_feats_raw = real_muons
        
        # Apply Normalization (from train.py parity)
        conditions = self.normalizer.normalize_primaries(prims)
        real_muons_feats = self.normalizer.normalize_features(real_muons_feats_raw)
            
        batch_size = conditions.size(0)
        device = conditions.device
        counts_cpu = counts.detach().cpu()

        # 1.5 Drop Empty Events (if requested)
        if self.hparams.drop_empty_events:
            empty_mask = counts_cpu == 0
            if empty_mask.any():
                keep_mask = ~empty_mask
                keep_events = torch.nonzero(keep_mask).flatten()
                
                if keep_events.numel() == 0:
                    return None
                    
                # Index mapping
                old_to_new = torch.full((counts.numel(),), -1, dtype=torch.long)
                old_to_new[keep_mask] = torch.arange(keep_events.numel(), dtype=torch.long)
                
                # Filter muons
                mu_keep = keep_mask[real_batch_idx.cpu()]
                real_muons_feats = real_muons_feats[mu_keep]
                real_batch_idx = old_to_new[real_batch_idx.cpu()][mu_keep].to(device)
                
                # Filter conditions/counts
                conditions = conditions[keep_mask]
                counts = counts[keep_mask]
                batch_size = conditions.size(0)
                counts_cpu = counts.detach().cpu()

        # 2. Outlier Filtering & Writing
        max_muons = self.hparams.max_muons_per_batch
        max_event_muons = self.hparams.max_muons_per_event
        event_muon_limit = max_event_muons if max_event_muons > 0 else max_muons
        
        counts_cpu = counts.detach().cpu()
        
        if event_muon_limit > 0:
            oversize_mask = counts_cpu > event_muon_limit
            if oversize_mask.any():
                oversize_idx = torch.nonzero(oversize_mask).flatten().tolist()
                
                if self.outlier_writer is not None:
                    for ev in oversize_idx:
                        c = int(counts_cpu[ev])
                        m_evt = (real_batch_idx == ev)
                        mu_evt = real_muons[m_evt]
                        self.outlier_writer.write_event(
                            source_file="unknown", # Dataloader info not easily available here
                            source_file_index=0,
                            batch_index=batch_idx,
                            event_index=ev,
                            count=c,
                            primaries=prims[ev],
                            muons=mu_evt
                        )
                
                # Filter
                keep_mask = ~oversize_mask
                keep_events = torch.nonzero(keep_mask).flatten()
                
                if keep_events.numel() == 0:
                    return None
                
                old_to_new = torch.full((counts.numel(),), -1, dtype=torch.long)
                old_to_new[keep_events] = torch.arange(keep_events.numel(), dtype=torch.long)
                
                mu_keep = keep_mask[real_batch_idx.cpu()]
                real_muons_feats = real_muons_feats[mu_keep]
                real_batch_idx = old_to_new[real_batch_idx.cpu()][mu_keep].to(device)
                conditions = conditions[keep_mask]
                counts = counts[keep_mask]
                batch_size = conditions.size(0)

        # 3. Micro-batching logic
        total_muons = int(counts.sum().item())
        if max_muons > 0 and total_muons > max_muons:
            # Split into micro-batches of contiguous event ranges
            start_ev = 0
            while start_ev < batch_size:
                cum = 0
                end_ev = start_ev
                while end_ev < batch_size:
                    c = int(counts[end_ev])
                    if (end_ev > start_ev) and (cum + c > max_muons):
                        break
                    cum += c
                    end_ev += 1
                    if cum >= max_muons:
                        break
                
                if end_ev <= start_ev:
                    end_ev = start_ev + 1
                
                sub_counts = counts[start_ev:end_ev]
                sub_cond = conditions[start_ev:end_ev]
                m = (real_batch_idx >= start_ev) & (real_batch_idx < end_ev)
                sub_muons = real_muons_feats[m]
                sub_bidx = real_batch_idx[m] - start_ev
                
                if sub_muons.numel() > 0:
                    self._run_optimization_step(sub_muons, sub_bidx, sub_cond, sub_counts)
                
                start_ev = end_ev
        else:
            self._run_optimization_step(real_muons_feats, real_batch_idx, conditions, counts)
        
        return None # Manual optimization

    def _run_optimization_step(self, real_muons, real_batch_idx, conditions, counts):
        opt_c, opt_g = self.optimizers()
        batch_size = conditions.size(0)
        device = conditions.device
        
        # Track for histograms
        self.last_real_feats = real_muons

        # --- CRITIC UPDATE ---
        for ci in range(int(max(1, self.critic_steps))):
            with torch.no_grad():
                fake_muons_flat, fake_batch_idx = self.generator(conditions)
            
            real_score = self.critic(real_muons, real_batch_idx, conditions, batch_size)
            fake_score = self.critic(fake_muons_flat.detach(), fake_batch_idx, conditions, batch_size)
            
            # W-distance gap for logging
            w_gap = real_score.mean() - fake_score.mean()
            self.log("w_gap", w_gap, prog_bar=True)
            self.log("train/w_gap", w_gap)

            # --- Event-Aligned GP (Matching training/model.py exactly) ---
            apply_gp = (self.lambda_gp > 0) and ((ci % self.hparams.gp_every) == 0)
            gp = torch.tensor(0.0, device=device)
            
            if apply_gp:
                # Compute pairing indices without gradients (this is just bookkeeping)
                with torch.no_grad():
                    real_counts_ev = torch.bincount(real_batch_idx, minlength=batch_size)
                    fake_counts_ev = torch.bincount(fake_batch_idx, minlength=batch_size)
                    k_ev = torch.minimum(real_counts_ev, fake_counts_ev)

                    # Sort indices
                    real_sorted_idx = torch.argsort(real_batch_idx)
                    fake_sorted_idx = torch.argsort(fake_batch_idx)

                    real_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
                    fake_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
                    if batch_size > 1:
                        real_offsets[1:] = torch.cumsum(real_counts_ev[:-1], dim=0)
                        fake_offsets[1:] = torch.cumsum(fake_counts_ev[:-1], dim=0)

                    nonzero_events = (k_ev > 0).nonzero().squeeze(1)
                    
                    if nonzero_events.numel() > 0:
                        k_vals = k_ev[nonzero_events]
                        r_starts = real_offsets[nonzero_events]
                        f_starts = fake_offsets[nonzero_events]
                        
                        # Vectorized range building
                        # (Note: Small CPU loop for list of ranges is often faster than pure GPU for this specific pattern)
                        pair_idx_within = torch.cat([torch.arange(k, device=device) for k in k_vals.cpu().tolist()])
                        event_id_per_pair = torch.repeat_interleave(torch.arange(nonzero_events.numel(), device=device), k_vals)
                        
                        idx_r = real_sorted_idx[r_starts[event_id_per_pair] + pair_idx_within]
                        idx_f = fake_sorted_idx[f_starts[event_id_per_pair] + pair_idx_within]
                        batch_idx_sub = nonzero_events[event_id_per_pair]

                        # Subsampling
                        n_pairs = idx_r.numel()
                        target_n = n_pairs
                        if self.hparams.gp_max_pairs > 0:
                            target_n = min(target_n, self.hparams.gp_max_pairs)
                        if 0.0 < self.hparams.gp_sample_fraction < 1.0:
                            target_n = min(target_n, max(1, int(n_pairs * self.hparams.gp_sample_fraction)))
                        
                        if target_n < n_pairs:
                            perm = torch.randperm(n_pairs, device=device)[:target_n]
                            idx_r, idx_f, batch_idx_sub = idx_r[perm], idx_f[perm], batch_idx_sub[perm]

                        real_sub = real_muons[idx_r]
                        fake_sub = fake_muons_flat.detach()[idx_f]
                
                # Now compute GP WITH gradients (outside the no_grad context)
                if nonzero_events.numel() > 0:
                    gp = compute_gp_flat(self.critic, real_sub, fake_sub, batch_idx_sub, conditions, batch_size)

            c_loss = fake_score.mean() - real_score.mean() + self.lambda_gp * gp
            c_loss_accum = c_loss / self.hparams.grad_accum_steps
            
            self.manual_backward(c_loss_accum)
            
            # Step every accumulate_grad_batches (handled manually because manual_optimization=True)
            if (ci + 1) % self.hparams.grad_accum_steps == 0:
                if self.hparams.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.grad_clip_norm)
                opt_c.step()
                opt_c.zero_grad()
            
            self.log("c_loss", c_loss, prog_bar=True)
            self.log("train/c_loss", c_loss)

        # --- GENERATOR UPDATE ---
        # 1. Multiplicity Loss
        target_multiplicity = torch.log10(counts.float().unsqueeze(1) + 1.0)
        pred_multiplicity = self.generator.multiplicity_net(conditions)
        m_loss = nn.functional.mse_loss(pred_multiplicity, target_multiplicity)

        # 2. Generator Loss (Fool Critic)
        fake_muons_flat, fake_batch_idx = self.generator(conditions)
        self.last_fake_feats = fake_muons_flat # Save for histograms
        
        fake_score_G = self.critic(fake_muons_flat, fake_batch_idx, conditions, batch_size)
        g_loss_adv = -fake_score_G.mean()
        
        total_g_loss = (g_loss_adv + m_loss) / self.hparams.grad_accum_steps
        
        self.manual_backward(total_g_loss)
        
        if self.hparams.grad_accum_steps == 1: # Simplified for now, should ideally track steps
            if self.hparams.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.hparams.grad_clip_norm)
            opt_g.step()
            opt_g.zero_grad()
        
        self.log("g_loss", g_loss_adv, prog_bar=True)
        self.log("m_loss", m_loss, prog_bar=True)
        self.log("train/g_loss", g_loss_adv)
        self.log("train/m_loss", m_loss)
