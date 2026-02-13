import torch
import numpy as np
import os
import joblib

class LazyDataPreprocessor:
    """
    A wrapper around sklearn preprocessors to fit on a subset of data 
    (accumulated from a dataloader) and then transform tensors.
    """
    def __init__(self, method='standard', feature_indices=None):
        """
        Args:
            method (str): 'standard', 'power', 'quantile', 'minmax', 'maxabs', 'robust'
            feature_indices (list of int, optional): Indices of features to transform. 
                                                     If None, transforms all.
        """
        self.method = method
        self.feature_indices = feature_indices
        self.scaler = None
        
        try:
            from sklearn.preprocessing import (
                StandardScaler, 
                PowerTransformer, 
                QuantileTransformer,
                MinMaxScaler,
                MaxAbsScaler,
                RobustScaler
            )
        except ImportError:
            raise ImportError("scikit-learn is required. Please install it with: pip install scikit-learn")

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'power':
            self.scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        elif method == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal')
        elif method == 'minmax':
             self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif method == 'maxabs':
            self.scaler = MaxAbsScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    def fit(self, dataloader, num_batches=100, feature_extractor=None):
        """
        Accumulates data from dataloader and fits the scaler.
        
        Args:
            dataloader: PyTorch DataLoader
            num_batches: Number of batches to accumulate
            feature_extractor: logic to extract features. If None, assumes batch is flat features 
                               or tuple where first element is features. 
                               For MuonDataModule: lambda b: b[0] (flat_muons)
        """
        data_list = []
        print(f"Accumulating {num_batches} batches for fitting...")
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            if feature_extractor:
                features = feature_extractor(batch)
            else:
                # Default heuristic
                if isinstance(batch, (list, tuple)):
                    features = batch[0]
                else:
                    features = batch
            
            # Move to CPU numpy
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
                
            # Filter NaNs/Infs
            if np.isnan(features).any() or np.isinf(features).any():
                continue
                
            # If processing only specific columns
            if self.feature_indices is not None:
                features = features[:, self.feature_indices]
                
            data_list.append(features)
            
        full_data = np.concatenate(data_list, axis=0)
        print(f"Fitting {self.method} scaler on {full_data.shape[0]} samples...")
        self.scaler.fit(full_data)
        print("Fitting complete.")
        return self

    def partial_fit(self, dataloader, num_batches=1, feature_extractor=None):
        """
        Incrementally fits the scaler on batches without loading all data into memory.
        Only works for methods that support partial_fit (standard, minmax, maxabs).
        
        Args:
            dataloader: PyTorch DataLoader
            num_batches: Number of batches to process in this call
            feature_extractor: logic to extract features.
        """
        if not hasattr(self.scaler, 'partial_fit'):
            raise ValueError(f"Method '{self.method}' does not support partial_fit. Use fit() instead.")

        print(f"Partial fitting on {num_batches} batches...")
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            if feature_extractor:
                features = feature_extractor(batch)
            else:
                if isinstance(batch, (list, tuple)):
                    features = batch[0]
                else:
                    features = batch
            
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
                
            if np.isnan(features).any() or np.isinf(features).any():
                continue
                
            if self.feature_indices is not None:
                features = features[:, self.feature_indices]
                
            self.scaler.partial_fit(features)
            
        print(f"Partial fit step complete. Processed {i+1} batches.")
        return i + 1

    def transform(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [Batch, Features]
        Returns:
            torch.Tensor: Transformed tensor
        """
        device = x.device
        dtype = x.dtype
        x_np = x.detach().cpu().numpy()
        
        if self.feature_indices is not None:
            # Only transform selected columns
            x_sel = x_np[:, self.feature_indices]
            x_trans = self.scaler.transform(x_sel)
            x_np_out = x_np.copy()
            x_np_out[:, self.feature_indices] = x_trans
            return torch.from_numpy(x_np_out).to(device=device, dtype=dtype)
        else:
            x_trans = self.scaler.transform(x_np)
            return torch.from_numpy(x_trans).to(device=device, dtype=dtype)

    def inverse_transform(self, x):
        device = x.device
        dtype = x.dtype
        x_np = x.detach().cpu().numpy()
        
        if self.feature_indices is not None:
            x_sel = x_np[:, self.feature_indices]
            x_inv = self.scaler.inverse_transform(x_sel)
            x_np_out = x_np.copy()
            x_np_out[:, self.feature_indices] = x_inv
            return torch.from_numpy(x_np_out).to(device=device, dtype=dtype)
        else:
            x_inv = self.scaler.inverse_transform(x_np)
            return torch.from_numpy(x_inv).to(device=device, dtype=dtype)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")

    def load(self, path):
        self.scaler = joblib.load(path)
        print(f"Scaler loaded from {path}")
