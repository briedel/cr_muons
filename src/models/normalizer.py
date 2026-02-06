import torch
import numpy as np

class DataNormalizer:
    def __init__(self):
        """Standard normalizer for IceCube Muon GAN data."""
        pass

    def normalize_primaries(self, primaries):
        """
        Normalize event-level primary features.
        Input: [Batch, 4+] [E_GeV, Zenith_Rad, Mass_A, (optional), Depth_m]
        Output: [Batch, 4] Normalized [log10(E/1e6), cos(zenith), log10(mass), depth/1000]
        """
        if primaries.ndim != 2:
            return primaries
        
        # 1. Physics Transform
        # Normalize primary energy to log PeV energy
        log_E = torch.log10(primaries[:, 0]/1e6)
        # normalize to cos(zenith)
        cos_Z = torch.cos(primaries[:, 1])
        # normalize to log(mass primary)
        log_A = torch.log10(primaries[:, 2])
        
        if primaries.shape[1] >= 5:
            depth_m = primaries[:, 4]
        elif primaries.shape[1] == 4:
            depth_m = primaries[:, 3]
        else:
            depth_m = torch.zeros_like(log_E)
            
        depth = depth_m / 1000.

        return torch.stack([log_E, cos_Z, log_A, depth], dim=1)

    def normalize_features_xy(self, features):
        """
        Input: [Batch, 3] -> [E_mu_GeV, X_m, Y_m]
        Output: [log10(E/1e6), X/500, Y/500]
        """
        # 1. Physics Transform
        # Normalize muon energy to log PeV energy
        log_E = torch.log10(features[:, 0]/1e6 + 1e-10)
        # Normalize to a radius of 500 m around the shower axis
        X = features[:, 1] / 500.
        Y = features[:, 2] / 500.

        return torch.stack([log_E, X, Y], dim=1)
    
    def normalize_features_r(self, features):
        """
        Input: [Batch, 3] -> [E_mu_GeV, R_m]
        Output: [log10(E/1e6), R/1000]
        """
        # 1. Physics Transform
        # Normalize muon energy to log PeV energy
        log_E = torch.log10(features[:, 0]/1e6 + 1e-10)
        # Normalize to a radius of 1000 m around the shower axis
        R = features[:, 1] / 1000.

        return torch.stack([log_E, R], dim=1)

    def denormalize_features(self, features_norm):
        """
        Input: Normalized Network Output
        Output: Physical units [GeV, meters, meters]
        """
        E_GeV = torch.pow(10, features_norm[:, 0])*1e6
        X_m = features_norm[:, 1] * 500.
        Y_m = features_norm[:, 2] * 500.
        
        return torch.stack([E_GeV, X_m, Y_m], dim=1)
