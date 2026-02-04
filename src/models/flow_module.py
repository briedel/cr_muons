import pytorch_lightning as pl
import torch
import torch.nn as nn
import zuko
from torch.distributions import Distribution

class MuonFlow(pl.LightningModule):
    def __init__(self, cond_dim=4, feat_dim=3, hidden_dim=256, 
                 bins=10, transforms=3, mult_loss_weight=0.1, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
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
        
    def log_prob(self, x, conditions):
        """Returns log_prob of muons x given conditions."""
        return self.flow(conditions).log_prob(x)

    def training_step(self, batch, batch_idx):
        # batch structure from hdf5_dataset.py ragged_collate_fn:
        # flat_muons: [Total_Muons, feat_dim]
        # batch_idx: [Total_Muons] (maps muon to event index)
        # prims: [Batch_Size, cond_dim]
        # counts: [Batch_Size]
        flat_muons, batch_idx, prims, counts = batch
        
        if prims.numel() == 0:
            return None

        # 1. Flow Loss: -log P(x | cond)
        # Expand conditions to match flat_muons
        expanded_conds = prims[batch_idx]
        flow_log_prob = self.flow(expanded_conds).log_prob(flat_muons)
        flow_loss = -flow_log_prob.mean()
        
        # 2. Multiplicity Loss: MSE on log(N+1)
        pred_log_counts = self.multiplicity_net(prims).squeeze(-1)
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
            
        expanded_conds = prims[batch_idx]
        flow_log_prob = self.flow(expanded_conds).log_prob(flat_muons)
        flow_loss = -flow_log_prob.mean()
        
        pred_log_counts = self.multiplicity_net(prims).squeeze(-1)
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
            
        # 1. Sample Multiplicity
        pred_log_counts = self.multiplicity_net(prims).squeeze(-1)
        pred_counts = torch.expm1(pred_log_counts).round().long()
        pred_counts = torch.clamp(pred_counts, min=0)
        
        # 2. Sample Muons from Flow
        all_samples = []
        for i, count in enumerate(pred_counts):
            if count > 0:
                # flow(context).sample((n,))
                samples = self.flow(prims[i:i+1]).sample((count,)).squeeze(1)
                all_samples.append(samples)
            else:
                all_samples.append(torch.zeros((0, self.hparams.feat_dim), device=self.device))
                
        return all_samples

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
