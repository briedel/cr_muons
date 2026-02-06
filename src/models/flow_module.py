import pytorch_lightning as pl
import torch
import torch.nn as nn
import zuko
from torch.distributions import Distribution

class MuonFlow(pl.LightningModule):
    def __init__(self, cond_dim=4, feat_dim=3, hidden_dim=256, 
                 bins=10, transforms=3, mult_loss_weight=0.1, lr=1e-4,
                 chunk_size=4096, debug=False):
        super().__init__()
        self.save_hyperparameters()
        self.chunk_size = chunk_size if chunk_size > 0 else 4096
        self.debug = debug
        if self.debug:
            print("DEBUG: Normalization debugging enabled.")
        print(f"Using chunk size: {self.chunk_size}")
        
        # 1. Multiplicity Network: P(N | cond)
        # Predicts log(N+1) to handle large ranges and zero-counts
        self.multiplicity_net = nn.Sequential(
            nn.Linear(cond_dim, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1) 
        )
        
        # 2. Flow Network: P(muon_feat | cond)
        # Neural Spline Flow (NSF) for high-dimensional density estimation
        self.flow = zuko.flows.NSF(
            features=feat_dim,
            context=cond_dim,
            bins=bins,
            transforms=transforms,
            hidden_features=[hidden_dim] * 2
        )
        
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
        return self.flow(c).log_prob(x)

    def training_step(self, batch, batch_idx):
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
        
        if self.debug and batch_idx == 0:
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

        # 1. Flow Loss: -log P(x | cond)
        # Expand conditions to match flat_muons
        expanded_conds = conditions[batch_idx]
        flow_log_prob = self.log_prob(muons_norm, expanded_conds)
        flow_loss = -flow_log_prob.mean()
        
        # 2. Multiplicity Loss: MSE on log(N+1)
        pred_log_counts = self.multiplicity_net(conditions).squeeze(-1)
        target_log_counts = torch.log1p(counts.float())
        multiplicity_loss = nn.functional.mse_loss(pred_log_counts, target_log_counts)
        
        # 3. Combined Loss
        total_loss = flow_loss + self.hparams.mult_loss_weight * multiplicity_loss
        
        self.log("train_flow_loss", flow_loss, prog_bar=True)
        self.log("train_mult_loss", multiplicity_loss, prog_bar=True)
        self.log("train_loss", total_loss)
        
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
                # flow(context).sample((n,))
                # Samples are in normalized space
                samples_norm = self.flow(conditions[i:i+1]).sample((count,)).squeeze(1)
                
                # Denormalize
                samples_phys = self.normalizer.denormalize_features(samples_norm)
                all_samples.append(samples_phys)
            else:
                all_samples.append(torch.zeros((0, self.hparams.feat_dim), device=self.device))
                
        return all_samples

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
