import torch
import numpy as np

class BaseNormalizer:
    def __init__(self):
        """Base normalizer for IceCube Muon data."""
        pass

    def normalize_primaries(self, primaries):
        """
        Normalize event-level primary features.
        Input: [Batch, 4+] [E_GeV, Zenith_Rad, Mass_A, (optional), Depth_m]
        Output: [Batch, 4] Normalized [log10(E/1e6), cos(zenith), log10(mass), depth/1000]
        """
        if primaries.ndim != 2:
            return primaries

        # Debug: Check for NaNs in primaries
        if torch.isnan(primaries).any():
             # Replace NaNs
             primaries = torch.nan_to_num(primaries, nan=1.0)
        
        # 1. Physics Transform
        # Normalize primary energy to log PeV energy
        p_e = torch.clamp(primaries[:, 0], min=1e-3)
        log_E = torch.log10(p_e/1e6)
        
        # normalize to cos(zenith)
        cos_Z = torch.cos(primaries[:, 1])
        
        # normalize to log(mass primary)
        # Mass A >= 1. Clamp to 0.9 to be safe.
        p_a = torch.clamp(primaries[:, 2], min=0.9)
        log_A = torch.log10(p_a)
        
        if primaries.shape[1] >= 5:
            depth_m = primaries[:, 4]
        elif primaries.shape[1] == 4:
            depth_m = primaries[:, 3]
        else:
            depth_m = torch.zeros_like(log_E)
            
        depth = depth_m / 1000.

        return torch.stack([log_E, cos_Z, log_A, depth], dim=1)


class GANNormalizer(BaseNormalizer):
    def __init__(self):
        super().__init__()

    def normalize_features(self, features):
        """
        Input: [Batch, N] -> [Batch, N] Normalized
        Automatically handles 2D (E, R) or 3D (E, X, Y) or 4D (E, R, X, Y)
        """
        # Debug: Check for NaNs in raw input
        if torch.isnan(features).any():
            print(f"[GANNormalizer] Warning: Found {torch.isnan(features).sum()} NaNs in raw features input!")
            features = torch.nan_to_num(features, nan=1.0)

        # 1. Physics Transform
        e_gev = features[:, 0]
        e_gev = torch.clamp(e_gev, min=1e-3) 
        log_E = torch.log10(e_gev/1e6)
        # Shift to make typical values around 0-10 for better training stability
        log_E = (log_E + 5.0)/2.

        if features.shape[1] == 2:
            # Assume [E, R]
            R = torch.log(features[:, 1]**2.+ 0.125)
            R = (R - 5.) / 2.0
            return torch.stack([log_E, R], dim=1)
        
        elif features.shape[1] == 3:
            # Assume [E, X, Y]
            X = features[:, 1] / 100.0
            Y = features[:, 2] / 100.0
            return torch.stack([log_E, X, Y], dim=1)
        
        elif features.shape[1] == 4:
            # Assume [E, R, X, Y]
            R = torch.log(features[:, 1]**2.+ 0.125)
            R = (R - 5.0) / 2.0
            
            X = features[:, 2] / 100.0
            Y = features[:, 3] / 100.0
            return torch.stack([log_E, R, X, Y], dim=1)
            
        return features

    def denormalize_features(self, features_norm):
        """
        Input: Normalized Network Output
        Output: Physical units [GeV, meters, meters]
        """
        # Reverse operations from normalize_features
        log_E = features_norm[:, 0] * 2.0 - 5.0
        E_GeV = torch.pow(10, log_E)*1e6
        
        if features_norm.shape[1] == 2:
            log_R2 = features_norm[:, 1] * 2.0 + 5.0
            R_m = torch.sqrt(torch.exp(log_R2) - 0.125)
            return torch.stack([E_GeV, R_m], dim=1)
            
        elif features_norm.shape[1] == 3:
            X_m = features_norm[:, 1] * 100.0
            Y_m = features_norm[:, 2] * 100.0
            return torch.stack([E_GeV, X_m, Y_m], dim=1)

        elif features_norm.shape[1] == 4:
            log_R2 = features_norm[:, 1] * 2.0 + 5.0
            R_m = torch.sqrt(torch.exp(log_R2) - 0.125)
            
            X_m = features_norm[:, 2] * 100.0
            Y_m = features_norm[:, 3] * 100.0
            return torch.stack([E_GeV, R_m, X_m, Y_m], dim=1)

        return features_norm


class FlowNormalizer(BaseNormalizer):
    def __init__(self):
        super().__init__()

    def normalize_features(self, features):
        """
        Input: [Batch, N] -> [Batch, N] Normalized
        Target range roughly [-5, 5] for Neural Spline Flows.
        """
        # Debug: Check for NaNs in raw input
        if torch.isnan(features).any():
            print(f"[FlowNormalizer] Warning: Found {torch.isnan(features).sum()} NaNs in raw features input!")
            features = torch.nan_to_num(features, nan=1.0)

        # 1. Physics Transform
        e_gev = features[:, 0]
        e_gev = torch.clamp(e_gev, min=1e-3) 
        log_E = torch.log10(e_gev/1e6)
        # Shift to make typical values centered around 0
        # Original logic: (log_E + 5.0)/2 maps [-3, 3] + 5 -> [2, 8] / 2 -> [1, 4]
        # Let's keep consistent with GAN for now, but explicit duplication allows modification later
        log_E = (log_E + 5.0)/2.

        if features.shape[1] == 2:
            # Assume [E, R]
            R = torch.log(features[:, 1]**2.+ 0.125)
            R = (R - 5.) / 2.0
            return torch.stack([log_E, R], dim=1)
        
        elif features.shape[1] == 3:
            # Assume [E, X, Y]
            X = features[:, 1] / 100.0
            Y = features[:, 2] / 100.0
            return torch.stack([log_E, X, Y], dim=1)
        
        elif features.shape[1] == 4:
            # Assume [E, R, X, Y]
            R = torch.log(features[:, 1]**2.+ 0.125)
            R = (R - 5.0) / 2.0
            
            X = features[:, 2] / 100.0
            Y = features[:, 3] / 100.0
            return torch.stack([log_E, R, X, Y], dim=1)
            
        return features

    def denormalize_features(self, features_norm):
        """
        Input: Normalized Network Output
        Output: Physical units [GeV, meters, meters]
        """
        # Reverse operations from normalize_features
        log_E = features_norm[:, 0] * 2.0 - 5.0
        E_GeV = torch.pow(10, log_E)*1e6
        
        if features_norm.shape[1] == 2:
            log_R2 = features_norm[:, 1] * 2.0 + 5.0
            R_m = torch.sqrt(torch.exp(log_R2) - 0.125)
            return torch.stack([E_GeV, R_m], dim=1)
            
        elif features_norm.shape[1] == 3:
            X_m = features_norm[:, 1] * 100.0
            Y_m = features_norm[:, 2] * 100.0
            return torch.stack([E_GeV, X_m, Y_m], dim=1)

        elif features_norm.shape[1] == 4:
            log_R2 = features_norm[:, 1] * 2.0 + 5.0
            R_m = torch.sqrt(torch.exp(log_R2) - 0.125)
            
            X_m = features_norm[:, 2] * 100.0
            Y_m = features_norm[:, 3] * 100.0
            return torch.stack([E_GeV, R_m, X_m, Y_m], dim=1)

        return features_norm

# Alias for backward compatibility if needed, though we will update usages.
DataNormalizer = GANNormalizer
    #     # Normalize to a radius to ln(R^2 + 0.25) to avoid singularity at R=0 
    #     R = torch.log(features[:, 1]**2.+ 0.125)
    #     # Alternative normalization: R = features[:, 1] / 1000.
    #     # R = features[:, 1] / 1000.

    #     return torch.stack([log_E, R], dim=1)
    
    # def normalize_features(self, features): 
    #     if features.shape[1] == 3:
    #         return self.normalize_features_xy(features)
    #     elif features.shape[1] == 2:
    #         return self.normalize_features_r(features)
    #     else:
    #         raise ValueError(f"Unsupported feature dimension: {features.shape[1]}")


