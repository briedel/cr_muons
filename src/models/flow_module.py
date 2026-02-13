import pytorch_lightning as pl
import torch
import torch.nn as nn
import zuko
from torch.distributions import Distribution, StudentT, Independent
from .normalizer import FlowNormalizer

class MuonFlow(pl.LightningModule):
    def __init__(self, cond_dim=4, feat_dim=3, hidden_dim=256, 
                 bins=10, transforms=3, mult_loss_weight=0.1, lr=1e-4,
                 chunk_size=4096, debug=False, context_embedding_dim=128,
                 base_dist="normal", student_dof=5.0):
        super().__init__()
        self.save_hyperparameters()
        self.chunk_size = chunk_size if chunk_size > 0 else 4096
        self.debug = debug
        if self.debug:
            print("DEBUG: Normalization debugging enabled.")
        print(f"Using chunk size: {self.chunk_size}")

        # Physics normalization
        self.normalizer = FlowNormalizer()
        
        self.avg_loss = None # For calculating smoothed loss for the scheduler
        self.loss_history = [] # For calculating slope

        # 1. Context Embedding Network
        # Processes raw physical conditions (logE, cosZ, etc.) into a dense representation
        self.context_embedding_dim = context_embedding_dim
        self.context_net = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.context_embedding_dim),
            nn.ELU(),
            nn.Linear(self.context_embedding_dim, self.context_embedding_dim),
            nn.ELU()
        )

        # 2. Multiplicity Network: P(N | cond)
        # Predicts log(N+1) to handle large ranges and zero-counts
        self.multiplicity_net = nn.Sequential(
            nn.Linear(cond_dim, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1) 
        )
        
        # 3. Flow Network: P(muon_feat | cond)
        # Neural Spline Flow (NSF) for high-dimensional density estimation
        self.flow = zuko.flows.NSF(
            features=feat_dim,
            context=self.context_embedding_dim,
            bins=bins,
            transforms=transforms,
            hidden_features=[hidden_dim] * 2
        )
        
        # Handle custom base distribution (e.g. Student-T)
        if base_dist == "student-t":
            print(f"Using Student-T base distribution with dof={student_dof}")
            # Replace the standard normal base with a StudentT
            # We wrap it in a Module to ensure parameters (dof) move to device
            class StudentTBase(nn.Module):
                def __init__(self, df, dim):
                    super().__init__()
                    self.register_buffer("df", torch.tensor(float(df)))
                    self.register_buffer("loc", torch.zeros(dim))
                    self.register_buffer("scale", torch.ones(dim))
                
                def forward(self, c=None):
                    # Zuko expects a base(c) call returning a distribution
                    return Independent(StudentT(self.df, self.loc, self.scale), 1)

                def log_prob(self, x):
                    return self.forward().log_prob(x)

                def sample(self, shape=torch.Size()):
                    return self.forward().sample(shape)
            
            self.flow.base = StudentTBase(student_dof, feat_dim)
        
    def log_prob(self, x, conditions, chunk_size=None):
        """Returns log_prob of muons x given conditions, with optional chunking/checkpointing."""
        if chunk_size is None:
            chunk_size = self.chunk_size

        total = x.shape[0]
        if total <= chunk_size:
            return self._forward_log_prob(x, conditions)
        
        # Chunking loop
        log_prob_list = []
        for i in range(0, total, chunk_size):
            end = min(i + chunk_size, total)
            x_chunk = x[i:end]
            c_chunk = conditions[i:end]
            
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                # using use_reentrant=False is recommended for newer PyTorch versions
                log_p = checkpoint(self._forward_log_prob, x_chunk, c_chunk, use_reentrant=False)
            else:
                log_p = self._forward_log_prob(x_chunk, c_chunk)
            
            log_prob_list.append(log_p)
            
        return torch.cat(log_prob_list)

    def _forward_log_prob(self, x, c):
        # Embed context
        emb = self.context_net(c)
        return self.flow(emb).log_prob(x)

    def training_step(self, batch, batch_idx_arg):
        # batch structure from hdf5_dataset.py ragged_collate_fn:
        # flat_muons: [Total_Muons, feat_dim]
        # batch_idx: [Total_Muons] (maps muon to event index)
        # prims: [Batch_Size, cond_dim]
        # counts: [Batch_Size]
        flat_muons, batch_idx, prims, counts = batch
        
        if prims.numel() == 0:
            return None

        # Apply Normalization
        conditions = self.normalizer.normalize_primaries(prims)
        
        # Check against typical shapes handled by dataloader/normalizer
        # batch_idx_arg is the Lightning Step index, batch_idx variable is the Muon-to-Event mapping tensor
        if self.debug and batch_idx_arg == 0:
            print("\n" + "="*50)
            print("[DEBUG] Normalization Check (First 10 Primaries):")
            print("Index | Raw (E, Z, A, ...) | Normalized (logE, cosZ, logA)")
            for i in range(min(10, prims.shape[0])):
                raw_str = ", ".join([f"{x:.2e}" for x in prims[i].tolist()])
                norm_str = ", ".join([f"{x:.4f}" for x in conditions[i].tolist()])
                print(f"{i:4d}  | [{raw_str}] | [{norm_str}]")
            print("="*50 + "\n")

        # Check against typical shapes handled by dataloader/normalizer
        if flat_muons.shape[1] == 5 and self.hparams.feat_dim == 3:
             muons_raw = flat_muons[:, 2:]
        else:
             muons_raw = flat_muons
             
        muons_norm = self.normalizer.normalize_features(muons_raw)

        # Check for values outside spline bounds [-5, 5]
        # Zuko NSF default bounds are [-5, 5]. Values outside are passed through linearly.
        out_of_bounds = (muons_norm.abs() > 5.0)
        if out_of_bounds.any():
            pct_out = out_of_bounds.float().mean().item() * 100
            # Only print warning occasionally to avoid spam
            if batch_idx_arg % 100 == 0:
                max_val = muons_norm.abs().max().item()
                print(f"[WARNING] {pct_out:.2f}% of muon features are outside spline bounds [-5, 5] (Max: {max_val:.2f}). "
                      "These values will not be transformed by the splines.")

        # Log ranges every 100 batches
        if batch_idx_arg % 100 == 0:
            with torch.no_grad():
                c_min = conditions.min(dim=0)[0].tolist()
                c_max = conditions.max(dim=0)[0].tolist()
                m_min = muons_norm.min(dim=0)[0].tolist()
                m_max = muons_norm.max(dim=0)[0].tolist()
                
                # Format for cleaner output
                def fmt(lst): return "[" + ", ".join([f"{x:.2f}" for x in lst]) + "]"

                print(f"[Batch {batch_idx_arg}] Stats:\n"
                      f"  Cond Min: {fmt(c_min)}\n"
                      f"  Cond Max: {fmt(c_max)}\n"
                      f"  Muon Min: {fmt(m_min)}\n"
                      f"  Muon Max: {fmt(m_max)}")

        if self.debug and batch_idx_arg == 0:
            print("[DEBUG] Muon Normalization Check (Muons in first 10 primaries):")
            print("Event | Raw Muon (E, ...) | Normalized Muon")
            for i in range(min(10, prims.shape[0])):
                mask = (batch_idx == i)
                if not mask.any():
                    continue

                raw_m = muons_raw[mask]
                norm_m = muons_norm[mask]
                
                # Print up to 3 muons per event
                for j in range(min(3, raw_m.shape[0])):
                    r_str = ", ".join([f"{x:.2e}" for x in raw_m[j].tolist()])
                    n_str = ", ".join([f"{x:.4f}" for x in norm_m[j].tolist()])
                    print(f"Evt {i} | [{r_str}] | [{n_str}]")
            print("="*50 + "\n")

        # 1. Flow Loss: -log P(x | cond)
        # Expand conditions to match flat_muons
        expanded_conds = conditions[batch_idx]
        flow_log_prob = self.log_prob(muons_norm, expanded_conds)
        flow_loss = -flow_log_prob.mean()
        
        # 2. Multiplicity Loss: MSE on log(N+1)
        pred_log_counts = self.multiplicity_net(conditions).squeeze(-1)
        target_log_counts = torch.log1p(counts.float())
        multiplicity_loss = nn.functional.mse_loss(pred_log_counts, target_log_counts)

        # Monitor Embedding and Multiplicity Stats
        if batch_idx_arg % 20 == 0:
            with torch.no_grad():
                # Context Embedding statistics (run forward pass on unique conditions)
                ctx_emb = self.context_net(conditions)
                self.log("mon/ctx_mean", ctx_emb.mean(), prog_bar=True)
                self.log("mon/ctx_std", ctx_emb.std(), prog_bar=True)
                self.log("mon/ctx_norm", torch.norm(ctx_emb, dim=1).mean(), prog_bar=True)
                
                # Multiplicity statistics
                self.log("mon/mult_pred_mean", pred_log_counts.mean(), prog_bar=True)
                self.log("mon/mult_pred_std", pred_log_counts.std(), prog_bar=True)
                self.log("mon/mult_target_mean", target_log_counts.mean(), prog_bar=True)
                
                # Print explicit monitor stats for immediate debugging
                print(f"[Batch {batch_idx_arg}] Mon Stats: "
                      f"CtxMean={ctx_emb.mean():.3f} CtxStd={ctx_emb.std():.3f} "
                      f"MultPred={pred_log_counts.mean():.3f} MultTgt={target_log_counts.mean():.3f}")

        # 3. Combined Loss
        total_loss = flow_loss + self.hparams.mult_loss_weight * multiplicity_loss
        
        # Smoothed loss for scheduler
        if self.avg_loss is None:
            self.avg_loss = total_loss.item()
            self.loss_history = [total_loss.item()]
        else:
            self.avg_loss = 0.95 * self.avg_loss + 0.05 * total_loss.item()
            self.loss_history.append(self.avg_loss)
            
        # Keep history up to 10k steps
        max_history = 10000
        if len(self.loss_history) > max_history:
            self.loss_history.pop(0)

        # Calculate slopes periodically
        if batch_idx_arg % 20 == 0:
            history_tensor = torch.tensor(self.loss_history, dtype=torch.float32)
            current_len = len(history_tensor)
            
            def calc_slope(y):
                n = len(y)
                if n < 2: return 0.0
                x = torch.arange(n, dtype=torch.float32)
                sum_x = torch.sum(x)
                sum_y = torch.sum(y)
                sum_xy = torch.sum(x * y)
                sum_xx = torch.sum(x * x)
                denom = (n * sum_xx - sum_x**2)
                if denom == 0: return 0.0
                return ((n * sum_xy - sum_x * sum_y) / denom).item()

            # Window 1: Short term (100 steps)
            if current_len >= 100:
                s_100 = calc_slope(history_tensor[-100:])
                self.log("slope_100", s_100 * 1000.0, prog_bar=True)

            # Window 2: Medium term (5000 steps)
            if current_len >= 5000:
                s_5k = calc_slope(history_tensor[-5000:])
                self.log("slope_5k", s_5k * 1000.0, prog_bar=True)
            
            # Window 3: Long term (10000 steps)
            if current_len >= 10000:
                s_10k = calc_slope(history_tensor[-10000:])
                self.log("slope_10k", s_10k * 1000.0, prog_bar=True)

        self.log("train_flow_loss", flow_loss, prog_bar=True)
        self.log("train_mult_loss", multiplicity_loss, prog_bar=True)
        self.log("train_loss", total_loss)
        self.log("train_loss_smooth", self.avg_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        flat_muons, batch_idx, prims, counts = batch
        if prims.numel() == 0:
            return None

        # Apply Normalization
        conditions = self.normalizer.normalize_primaries(prims)
        
        if flat_muons.shape[1] == 5 and self.hparams.feat_dim == 3:
             muons_raw = flat_muons[:, 2:]
        else:
             muons_raw = flat_muons
             
        muons_norm = self.normalizer.normalize_features(muons_raw)
            
        expanded_conds = conditions[batch_idx]
        flow_log_prob = self.log_prob(muons_norm, expanded_conds)
        flow_loss = -flow_log_prob.mean()
        
        pred_log_counts = self.multiplicity_net(conditions).squeeze(-1)
        target_log_counts = torch.log1p(counts.float())
        multiplicity_loss = nn.functional.mse_loss(pred_log_counts, target_log_counts)
        
        val_loss = flow_loss + self.hparams.mult_loss_weight * multiplicity_loss
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # We expect conditions (prims) in the batch
        if isinstance(batch, (list, tuple)):
            prims = batch[2] if len(batch) > 2 else batch[0]
        else:
            prims = batch
            
        # Normalize conditions
        conditions = self.normalizer.normalize_primaries(prims)

        # 1. Sample Multiplicity
        pred_log_counts = self.multiplicity_net(conditions).squeeze(-1)
        pred_counts = torch.expm1(pred_log_counts).round().long()
        pred_counts = torch.clamp(pred_counts, min=0)
        
        # 2. Sample Muons from Flow
        all_samples = []
        for i, count in enumerate(pred_counts):
            if count > 0:
                # Embed context for this event
                emb = self.context_net(conditions[i:i+1])
                # flow(context).sample((n,))
                # Samples are in normalized space
                samples_norm = self.flow(emb).sample((count,)).squeeze(1)
                
                # Denormalize
                samples_phys = self.normalizer.denormalize_features(samples_norm)
                all_samples.append(samples_phys)
            else:
                all_samples.append(torch.zeros((0, self.hparams.feat_dim), device=self.device))
                
        return all_samples

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2000,
            threshold=1e-4,
            min_lr=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_smooth",
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["avg_loss"] = self.avg_loss
        checkpoint["loss_history"] = self.loss_history

    def on_load_checkpoint(self, checkpoint):
        self.avg_loss = checkpoint.get("avg_loss", None)
        self.loss_history = checkpoint.get("loss_history", [])
