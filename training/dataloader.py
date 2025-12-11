import torch
import h5py
import numpy as np
import bisect

class SingleHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # Open file in read mode (SWMR is safer for concurrent reading)
        self.h5_file = h5py.File(h5_path, 'r', swmr=True)
        self.primaries = self.h5_file["primaries"]
        self.muons = self.h5_file["muons"]
        self.counts = self.h5_file["counts"]
        
        # Pre-load 'counts' into RAM (it's small) to build the index map faster
        self.counts_ram = self.counts[:] 
        
        # Pre-calculate cumulative sum to know where each event starts in the muon list
        self.muon_offsets = np.concatenate(([0], np.cumsum(self.counts_ram)[:-1]))

    def __getitem__(self, idx):
        # 1. Get Condition
        prim = torch.tensor(self.primaries[idx])
        
        # 2. Find Muons for this event
        start = self.muon_offsets[idx]
        end = start + self.counts_ram[idx]
        
        # 3. Slice directly from disk (Fast if chunks are well-sized)
        if start == end:
             # Handle empty event (zero muons)
             muons = torch.zeros((0, 3))
        else:
             muons = torch.tensor(self.muons[start:end])

        if not torch.equal(torch.unique(prim[0:2]), 
                           torch.unique(muons[:,0:2])):
            raise RuntimeError("Major and Minor IDs for primaries and muons don't match")

        # Do not return the primary major and minor id
        return prim[2:], muons[:,2:]

    def __len__(self):
        return len(self.counts_ram)

class MultiHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        """
        Args:
            file_paths (list): List of paths to .h5 files
                               e.g. ["data_0.h5", "data_1.h5"]
        """
        self.file_paths = sorted(file_paths)
        self.files = [None] * len(self.file_paths) # Lazy loading handles
        
        # 1. Build Global Index
        # We need to know how many events are in each file to build a map.
        # We read just the 'shape' metadata (very fast).
        self.file_offsets = [0]
        self.total_events = 0
        
        print("Scanning HDF5 files...")
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                n_events = f["counts"].shape[0] # Number of events in this file
                self.total_events += n_events
                self.file_offsets.append(self.total_events)
        
        # file_offsets is now [0, N_file1, N_file1+N_file2, ...]
        print(f"Total Events Found: {self.total_events}")

    def _open_file(self, file_idx):
        """Opens file handle if not already open."""
        if self.files[file_idx] is None:
            # swmr=True (Single Writer Multiple Reader) is safer for concurrency
            self.files[file_idx] = h5py.File(
                self.file_paths[file_idx], 'r', swmr=True)
            
            # Pre-load 'counts' for this file into RAM for speed
            # (Counts array is small: 400k ints = 1.6 MB)
            self.files[file_idx].counts_cache = self.files[
                file_idx]["counts"][:]
            
            # Build local muon offsets for this file
            # If counts=[2, 3], offsets=[0, 2, 5]
            local_counts = self.files[file_idx].counts_cache
            self.files[file_idx].muon_offsets = np.concatenate(([0], 
                np.cumsum(local_counts)[:-1]))
            
        return self.files[file_idx]

    def __getitem__(self, global_idx):
        # 1. Find which file this global_idx belongs to
        # 'bisect_right' gives insertion point. -1 gives us the file index.
        file_idx = bisect.bisect_right(self.file_offsets, global_idx) - 1
        
        # 2. Calculate local index within that file
        local_idx = global_idx - self.file_offsets[file_idx]
        
        # 3. Access the file
        f = self._open_file(file_idx)
        
        # 4. Get Condition
        prim = torch.tensor(f["conditions"][local_idx])
        
        # 5. Get Muons (using cached offsets)
        start = f.muon_offsets[local_idx]
        count = f.counts_cache[local_idx]
        end = start + count
        
        if count == 0:
            muons = torch.zeros((0, 3), dtype=torch.float32)
        else:
            muons = torch.tensor(f["muons"][start:end])

        # This is a sanity check to make sure the primary major and minor ids
        # for the primary and the primary of the muons is the same
        if not torch.equal(torch.unique(prim[0:2]), 
                           torch.unique(muons[:,0:2])):
            raise RuntimeError("Major and Minor IDs for primaries and muons don't match")
        
        # Do not return the primary major and minor id 
        # that are the first two entries for each primary and muon
        return prim[2:], muons[:,2:]

    def __len__(self):
        return self.total_events
    
    def __del__(self):
        """Cleanup file handles on exit"""
        for f in self.files:
            if f is not None:
                f.close()

def ragged_collate_fn(batch):
    """
    Custom collate function to handle variable-length muon bundles.
    
    Args:
        batch: List of tuples (cond, muons) returned by __getitem__
               cond shape: [4]
               muons shape: [N_i, 3] (variable N_i)
    """
    
    # 1. Stack Conditions (These are fixed size, so stack works)
    # Result shape: [Batch_Size, 4]
    prims = torch.stack([item[0] for item in batch])
    
    # 2. Extract Muon tensors
    muon_list = [item[1] for item in batch]
    
    # 3. Calculate Counts (How many muons in each event?)
    # Result shape: [Batch_Size]
    counts = torch.tensor([len(m) for m in muon_list], dtype=torch.long)
    
    # 4. Flatten Muons (Concatenate instead of Stack)
    # Result shape: [Total_Muons_In_Batch, 3]
    if len(muon_list) > 0:
        flat_muons = torch.cat(muon_list, dim=0)
    else:
        # Handle edge case of completely empty batch
        flat_muons = torch.empty((0, 3))
        
    # 5. Create Batch Index (Maps flat muons back to their event index)
    # Example: If counts is [2, 1, 3], batch_idx is [0, 0, 1, 2, 2, 2]
    batch_size = len(batch)
    batch_idx = torch.repeat_interleave(torch.arange(batch_size), counts)
    
    return flat_muons, batch_idx, prims, counts