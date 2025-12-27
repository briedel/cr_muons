import torch
import numpy as np

class DataNormalizer:
    def __init__(self, stats_dict=None):
        """
        stats_dict: Dictionary containing pre-computed constants 
                    (mean, std, min, max) for your dataset.
        """
        # Example stats - YOU MUST CALCULATE THESE FROM YOUR TRAINING DATA
        # if stats_dict is None:
        #     self.stats = {
        #         # [log10(E_prim), Cos(Zenith), log10(Mass), Depth]
        #         'cond_mean': torch.tensor([5.0, 0.7, 0.5, 3.0]), 
        #         'cond_std':  torch.tensor([2.0, 0.2, 0.5, 1.0]),
                
        #         # [log10(E_mu), X, Y]
        #         'feat_mean': torch.tensor([2.5, 0.0, 0.0]),
        #         'feat_std':  torch.tensor([1.5, 100.0, 100.0]), 
        #     }
        # else:
        #     self.stats = stats_dict

    def normalize_primaries(self, primaries):
        """
        Input: [Batch, 4] -> [E_GeV, Zenith_Rad, Mass_A, Depth_km]
        Output: [Batch, 4] Normalized
        """
        # Ensure input is on the correct device
        # primaries = primaries.to(self.device) # If we had self.device
        
        # 1. Physics Transform
        # Normalize primary energy to log PeV energy
        log_E = torch.log10(primaries[:, 0]/1e6)
        # normalize to cos(zenith)
        cos_Z = torch.cos(primaries[:, 1])
        # normalize to log(mass primary)
        log_A = torch.log10(primaries[:, 2])
        # primaries[:,3] is the relative time of the primary interaction
        # not used right now
        # normalize to km slant depth
        depth = primaries[:, 4] / 1000.

        return torch.stack([log_E, cos_Z, log_A, depth], dim=1)

    def normalize_features(self, features):
        """
        Input: [Batch, 3] -> [E_mu_GeV, X_m, Y_m]
        """
        # 1. Physics Transform
        # Normalize muon energy to log PeV energy
        # Add epsilon to energy to avoid log(0)
        log_E = torch.log10(features[:, 0]/1e6 + 1e-10)
        # Normalize to a radius of 500 m around the shower axis
        X = features[:, 1] / 500.
        Y = features[:, 2] / 500.

        # # 2. Z-Score
        # x_norm = (x - self.stats['feat_mean'].to(x.device)) / self.stats['feat_std'].to(x.device)
        # return x_norm
        return torch.stack([log_E, X, Y], dim=1)

    def denormalize_features(self, features_norm):
        """
        Input: Normalized Network Output
        Output: Physical units [GeV, meters, meters]
        """
        # device = features_norm.device
        
        # # 1. Inverse Z-Score
        # # x = x_norm * std + mean
        # x = features_norm * self.stats['feat_std'].to(device) + self.stats['feat_mean'].to(device)
        
        # 2. Inverse Physics Transform
        E_GeV = torch.pow(10, features_norm[:, 0])*1e6
        X_m = features_norm[:, 1] * 500.
        Y_m = features_norm[:, 2] * 500.
        
        return torch.stack([E_GeV, X_m, Y_m], dim=1)