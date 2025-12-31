import torch
import numpy as np
import h5py
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    from datasets import IterableDataset, Features, Sequence, Value
except ImportError:
    print("Hugging Face datasets library not found. Install with `pip install datasets`")

def process_h5_file(f):
    """Helper to yield examples from an open h5py File object"""
    primaries = f['primaries']
    muons = f['muons']
    counts = f['counts']
    
    # Read counts to memory to calculate offsets
    local_counts = counts[:]
    
    current_offset = 0
    
    for i, count in enumerate(local_counts):
        # Get primary (keeping IDs)
        p = primaries[i]
        
        # Get muons
        start = current_offset
        end = current_offset + count
        current_offset = end
        
        if count == 0:
            m = np.zeros((0, 5), dtype=np.float32)
        else:
            m = muons[start:end]
        
        yield {
            "primary": p,
            "muons": m
        }

def h5_generator(file_paths, federation_url=None, token=None):
    """
    Generator that yields examples from HDF5 files.
    This allows streaming data into a Hugging Face Dataset.
    Supports Pelican FS if federation_url is provided.
    """
    # Ensure paths are strings
    file_paths = [str(p) for p in file_paths]
    
    fs = None
    if federation_url:
        try:
            from pelicanfs.core import PelicanFileSystem
        except ImportError:
            raise ImportError("pelicanfs is required. Install with: pip install pelicanfs")
        
        headers = None
        if token:
            headers = {f"Authorization": f"Bearer {token}"}
        
        fs = PelicanFileSystem(federation_url, headers=headers)
    
    for path in file_paths:
        if fs:
            # Open remote file via Pelican
            # We use a context manager to ensure the remote file handle is closed
            with fs.open(path, 'rb') as remote_f:
                with h5py.File(remote_f, 'r') as f:
                    yield from process_h5_file(f)
        else:
            # Open local file
            with h5py.File(path, 'r') as f:
                yield from process_h5_file(f)

def parquet_generator(file_paths, federation_url=None, token=None):
    """
    Generator that yields examples from Parquet files.
    """
    if pq is None:
        raise ImportError("pyarrow is required to read parquet files. Install with `pip install pyarrow`")

    # Ensure paths are strings
    file_paths = [str(p) for p in file_paths]
    
    fs = None
    if federation_url:
        try:
            from pelicanfs.core import PelicanFileSystem
        except ImportError:
            raise ImportError("pelicanfs is required. Install with: pip install pelicanfs")
        
        headers = None
        if token:
            headers = {f"Authorization": f"Bearer {token}"}
        
        fs = PelicanFileSystem(federation_url, headers=headers)
    
    for path in file_paths:
        if fs:
            # Open remote file via Pelican
            with fs.open(path, 'rb') as remote_f:
                pf = pq.ParquetFile(remote_f)
                for i in range(pf.num_row_groups):
                    table = pf.read_row_group(i)
                    pydict = table.to_pydict()
                    has_id_cols = (
                        'primary_major_id' in pydict and
                        'primary_minor_id' in pydict
                    )

                    if has_id_cols:
                        for maj, minr, p, m in zip(
                            pydict['primary_major_id'],
                            pydict['primary_minor_id'],
                            pydict['primary'],
                            pydict['muons'],
                        ):
                            maj_f = float(np.int64(maj))
                            min_f = float(np.int64(minr))
                            p_out = [maj_f, min_f] + list(p)
                            m_out = [[maj_f, min_f] + list(row) for row in m]
                            yield {"primary": p_out, "muons": m_out}
                    else:
                        for p, m in zip(pydict['primary'], pydict['muons']):
                            yield {"primary": p, "muons": m}
        else:
            pf = pq.ParquetFile(path)
            for i in range(pf.num_row_groups):
                table = pf.read_row_group(i)
                pydict = table.to_pydict()
                has_id_cols = (
                    'primary_major_id' in pydict and
                    'primary_minor_id' in pydict
                )

                if has_id_cols:
                    for maj, minr, p, m in zip(
                        pydict['primary_major_id'],
                        pydict['primary_minor_id'],
                        pydict['primary'],
                        pydict['muons'],
                    ):
                        maj_f = float(np.int64(maj))
                        min_f = float(np.int64(minr))
                        p_out = [maj_f, min_f] + list(p)
                        m_out = [[maj_f, min_f] + list(row) for row in m]
                        yield {"primary": p_out, "muons": m_out}
                else:
                    for p, m in zip(pydict['primary'], pydict['muons']):
                        yield {"primary": p, "muons": m}

def get_hf_dataset(file_paths, file_format='h5', streaming=True, federation_url=None, token=None):
    """
    Creates a Hugging Face Dataset from HDF5 or Parquet files.
    
    Args:
        file_paths: List of paths to files
        file_format: 'h5' or 'parquet'
        streaming: If True, returns an IterableDataset (lazy loading)
        federation_url: Optional Pelican federation URL
        token: Optional auth token
    """
    # Define features to ensure correct types and shapes
    # primary: [4] (log10(E), cos(theta), log10(A), depth)
    # muons: [N, 3] (log10(E), X, Y)
    features = Features({
        "primary": Sequence(Value("float32")),
        "muons": Sequence(Sequence(Value("float32"))) 
    })
    
    if file_format == 'h5':
        gen = h5_generator
    elif file_format == 'parquet':
        gen = parquet_generator
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    ds = IterableDataset.from_generator(
        gen, 
        gen_kwargs={
            "file_paths": file_paths,
            "federation_url": federation_url,
            "token": token
        }, 
        features=features
    )
    
    return ds

def hf_collate_fn(batch):
    """
    Custom collate function for Hugging Face Dataset batches.
    Adapts the dictionary format to the flat tensor format required by the model.
    
    Args:
        batch: List of dicts [{'primary': [...], 'muons': [[...], ...]}, ...]
        
    Returns:
        flat_muons: [Total_Muons, 3]
        batch_idx: [Total_Muons]
        prims: [Batch_Size, 4]
        counts: [Batch_Size]
    """
    # 1. Stack Primaries
    prims = torch.stack([torch.tensor(item['primary']) for item in batch])
    
    # 2. Process Muons
    # Convert lists to tensors
    muon_list = [torch.tensor(item['muons']) for item in batch]
    
    # 3. Counts
    counts = torch.tensor([len(m) for m in muon_list], dtype=torch.long)
    
    # 4. Flatten Muons
    if len(muon_list) > 0:
        # Handle empty events correctly (tensor with shape (0, 3))
        valid_muons = [m for m in muon_list if m.numel() > 0]
        if valid_muons:
            flat_muons = torch.cat(valid_muons, dim=0)
        else:
            flat_muons = torch.empty((0, 3))
    else:
        flat_muons = torch.empty((0, 3))
        
    # 5. Create Batch Index
    batch_size = len(batch)
    batch_idx = torch.repeat_interleave(torch.arange(batch_size), counts)
    
    return flat_muons, batch_idx, prims, counts
