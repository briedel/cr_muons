import torch
import numpy as np
import h5py
import io
from collections import OrderedDict
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    from datasets import IterableDataset, Features, Sequence, Value
except ImportError:
    print("Hugging Face datasets library not found. Install with `pip install datasets`")

try:
    from torch.utils.data import IterableDataset as TorchIterableDataset
except Exception:
    TorchIterableDataset = object

from .parquet_dataset import FileBytesLRUCache


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

def h5_generator(file_paths, federation_url=None, token=None, memory_cache=None):
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

def parquet_generator(file_paths, federation_url=None, token=None, memory_cache=None):
    """Generator that yields examples from Parquet files.

    If `memory_cache` is provided (e.g. FileBytesLRUCache), this will cache whole
    Parquet file bytes in RAM up to the configured limit and read via BytesIO.
    """
    if pq is None:
        raise ImportError("pyarrow is required to read parquet files. Install with `pip install pyarrow`")

    file_paths = [str(p) for p in file_paths]

    fs = None
    if federation_url:
        try:
            from pelicanfs.core import PelicanFileSystem
        except ImportError:
            raise ImportError("pelicanfs is required. Install with: pip install pelicanfs")

        headers = {"Authorization": f"Bearer {token}"} if token else None
        fs = PelicanFileSystem(federation_url, headers=headers)

    def _normalize_muons(m, *, default_dim: int = 3):
        if m is None:
            return np.zeros((0, default_dim), dtype=np.float32)

        if isinstance(m, list):
            if len(m) == 0:
                return np.zeros((0, default_dim), dtype=np.float32)
            first = m[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                inferred_dim = len(first)
                arr = np.asarray(m, dtype=np.float32)
                return arr.reshape((-1, inferred_dim))

        arr = np.asarray(m, dtype=np.float32)
        if arr.ndim == 1 and arr.size == 0:
            return np.zeros((0, default_dim), dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape((-1, default_dim))
        return arr

    def _iter_parquet_file(pf):
        """Iterate a parquet file via Arrow record batches.

        This avoids `table.to_pydict()` (which materializes full Python lists)
        and instead walks smaller Arrow record batches.
        """
        try:
            import pyarrow as pa
        except Exception:
            pa = None

        # Tuneable chunk size: larger reduces Python overhead, but uses more RAM.
        record_batch_size = 4096

        # Only read the columns we need.
        columns = ["primary", "muons"]

        for rb in pf.iter_batches(batch_size=record_batch_size, columns=columns):
            n = rb.num_rows
            if n <= 0:
                continue

            prim_col = rb.column(0)
            mu_col = rb.column(1)

            # Fast path: fixed-size primary vectors -> numpy 2D (fewer Python objects).
            prim_np = None
            prim_list_size = None
            if pa is not None:
                try:
                    if pa.types.is_fixed_size_list(prim_col.type):
                        prim_list_size = prim_col.type.list_size
                        # Flatten values then reshape. This may copy depending on Arrow layout,
                        # but still avoids per-row Python list creation for primaries.
                        prim_np = prim_col.values.to_numpy(zero_copy_only=False).reshape((n, prim_list_size))
                except Exception:
                    prim_np = None

            for i in range(n):
                if prim_np is not None:
                    p = prim_np[i]
                else:
                    p = prim_col[i].as_py()
                m = mu_col[i].as_py()
                yield {"primary": p, "muons": _normalize_muons(m, default_dim=3)}

    cache_enabled = bool(memory_cache is not None and getattr(memory_cache, "enabled", lambda: False)())

    for path in file_paths:
        if fs:
            if cache_enabled:
                def _load_bytes() -> bytes:
                    with fs.open(path, "rb") as remote_f:
                        return remote_f.read()

                raw = memory_cache.get_or_load(path, _load_bytes)
                pf = pq.ParquetFile(io.BytesIO(raw))
                yield from _iter_parquet_file(pf)
            else:
                with fs.open(path, "rb") as remote_f:
                    pf = pq.ParquetFile(remote_f)
                    yield from _iter_parquet_file(pf)
        else:
            if cache_enabled:
                def _load_bytes() -> bytes:
                    with open(path, "rb") as f:
                        return f.read()

                raw = memory_cache.get_or_load(path, _load_bytes)
                pf = pq.ParquetFile(io.BytesIO(raw))
                yield from _iter_parquet_file(pf)
            else:
                pf = pq.ParquetFile(path)
                yield from _iter_parquet_file(pf)

def get_hf_dataset(file_paths, file_format='h5', streaming=True, federation_url=None, token=None, memory_cache=None):
    """
    Creates a Hugging Face Dataset from HDF5 or Parquet files.
    
    Args:
        file_paths: List of paths to files
        file_format: 'h5' or 'parquet'
        streaming: If True, returns an IterableDataset (lazy loading)
        federation_url: Optional Pelican federation URL
        token: Optional auth token
        memory_cache: Optional FileBytesLRUCache for caching whole-file bytes
    """
    # Define features to ensure correct types and shapes.
    # Note: IDs may exist in Parquet as separate columns, but we intentionally
    # do NOT include them in the streamed dataset or training tensors.
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
            "token": token,
            "memory_cache": memory_cache,
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
    # Use as_tensor to avoid unnecessary copies when inputs are already numpy arrays.
    prims = torch.stack([torch.as_tensor(item["primary"], dtype=torch.float32) for item in batch])
    
    # 2. Process Muons
    # Convert lists/arrays to tensors (prefer as_tensor to avoid copies)
    muon_list = [torch.as_tensor(item["muons"], dtype=torch.float32) for item in batch]
    
    # 3. Counts
    counts = torch.tensor([int(m.shape[0]) if m.ndim >= 1 else int(m.numel()) for m in muon_list], dtype=torch.long)
    
    # 4. Flatten Muons
    if len(muon_list) > 0:
        # Handle empty events correctly (tensor with shape (0, 3))
        valid_muons = [m for m in muon_list if m.numel() > 0]
        if valid_muons:
            flat_muons = torch.cat(valid_muons, dim=0)
        else:
            flat_muons = torch.empty((0, 3), dtype=torch.float32)
    else:
        flat_muons = torch.empty((0, 3), dtype=torch.float32)
        
    # 5. Create Batch Index
    batch_size = len(batch)
    batch_idx = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.long), counts)
    
    return flat_muons, batch_idx, prims, counts
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
    # Use as_tensor to avoid unnecessary copies when inputs are already numpy arrays.
    prims = torch.stack([torch.as_tensor(item["primary"], dtype=torch.float32) for item in batch])
    
    # 2. Process Muons
    # Convert lists/arrays to tensors (prefer as_tensor to avoid copies)
    muon_list = [torch.as_tensor(item["muons"], dtype=torch.float32) for item in batch]
    
    # 3. Counts
    counts = torch.tensor([int(m.shape[0]) if m.ndim >= 1 else int(m.numel()) for m in muon_list], dtype=torch.long)
    
    # 4. Flatten Muons
    if len(muon_list) > 0:
        # Handle empty events correctly (tensor with shape (0, 3))
        valid_muons = [m for m in muon_list if m.numel() > 0]
        if valid_muons:
            flat_muons = torch.cat(valid_muons, dim=0)
        else:
            flat_muons = torch.empty((0, 3), dtype=torch.float32)
    else:
        flat_muons = torch.empty((0, 3), dtype=torch.float32)
        
    # 5. Create Batch Index
    batch_size = len(batch)
    batch_idx = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.long), counts)
    
    return flat_muons, batch_idx, prims, counts
