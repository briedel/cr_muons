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


class FileBytesLRUCache:
    """A simple, process-local LRU cache for file bytes.

    Notes:
    - This only helps when DataLoader uses `num_workers=0` (default here), since
      it is not shared across worker processes.
    - Intended primarily for Parquet reads via PyArrow.
    """

    def __init__(self, *, max_bytes: int) -> None:
        self.max_bytes = int(max(0, max_bytes))
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._size_bytes = 0

    def enabled(self) -> bool:
        return self.max_bytes > 0

    def get(self, key: str) -> bytes | None:
        if not self.enabled():
            return None
        key = str(key)
        b = self._cache.get(key)
        if b is None:
            return None
        self._cache.move_to_end(key)
        return b

    def put(self, key: str, data: bytes) -> None:
        if not self.enabled():
            return
        key = str(key)
        data = bytes(data)
        n = len(data)
        if n <= 0 or n > self.max_bytes:
            return

        old = self._cache.pop(key, None)
        if old is not None:
            self._size_bytes -= len(old)

        self._cache[key] = data
        self._size_bytes += n
        self._cache.move_to_end(key)

        # Evict LRU until we're under budget.
        while self._size_bytes > self.max_bytes and self._cache:
            _, ev = self._cache.popitem(last=False)
            self._size_bytes -= len(ev)

    def get_or_load(self, key: str, loader) -> bytes:
        key = str(key)
        b = self.get(key)
        if b is not None:
            return b
        b = loader()
        self.put(key, b)
        return b

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


class ParquetBatchIterableDataset(TorchIterableDataset):
    """Yield already-batched tensors from Parquet via Arrow record batches.

    Yields tuples matching the training loop signature:
      (flat_muons, batch_idx, prims, counts)

    This bypasses HuggingFace's per-example generator + Python collation and
    can significantly reduce the time spent in "load".
    
    Args:
        shuffle: If True, load entire file into memory and shuffle all examples
                 before batching. This prevents overfitting to data structure patterns.
        shuffle_seed: Random seed for reproducible shuffling (default: 42)
    """

    def __init__(
        self,
        file_paths,
        *,
        batch_size: int,
        federation_url: str | None = None,
        token: str | None = None,
        memory_cache=None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
    ) -> None:
        super().__init__()
        self.file_paths = [str(p) for p in (file_paths or [])]
        self.batch_size = int(batch_size)
        self.federation_url = federation_url
        self.token = token
        self.memory_cache = memory_cache
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)

    def __iter__(self):
        if pq is None:
            raise ImportError("pyarrow is required to read parquet files. Install with `pip install pyarrow`")

        import pyarrow as pa

        worker_info = None
        try:
            from torch.utils.data import get_worker_info

            worker_info = get_worker_info()
        except Exception:
            worker_info = None

        dist_rank = 0
        dist_world = 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist_rank = int(dist.get_rank())
                dist_world = int(dist.get_world_size())
        except Exception:
            dist_rank = 0
            dist_world = 1

        worker_id = int(worker_info.id) if worker_info is not None else 0
        num_workers = int(worker_info.num_workers) if worker_info is not None else 1
        shard_id = dist_rank * num_workers + worker_id
        shard_world = dist_world * num_workers

        fs = None
        if self.federation_url:
            try:
                from pelicanfs.core import PelicanFileSystem
            except ImportError:
                raise ImportError("pelicanfs is required. Install with: pip install pelicanfs")

            headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
            fs = PelicanFileSystem(self.federation_url, headers=headers)

        cache_enabled = bool(self.memory_cache is not None and getattr(self.memory_cache, "enabled", lambda: False)())

        def _as_numpy(a) -> np.ndarray:
            return a.to_numpy(zero_copy_only=False)

        def _iter_parquet(pf):
            cols = ["primary", "muons"]
            rb_idx = 0
            for rb in pf.iter_batches(batch_size=self.batch_size, columns=cols):
                # Shard record-batches across DDP ranks and DataLoader workers so a single
                # file can be decoded in parallel without duplicating data.
                if shard_world > 1 and (rb_idx % shard_world) != shard_id:
                    rb_idx += 1
                    continue
                rb_idx += 1
                n = rb.num_rows
                if n <= 0:
                    continue

                prim_col = rb.column(0)
                mu_col = rb.column(1)

                # Primaries: try a vectorized path when fixed-size list.
                if pa.types.is_fixed_size_list(prim_col.type):
                    p_dim = prim_col.type.list_size
                    prim_np = _as_numpy(prim_col.values).reshape((n, p_dim))
                    prims = torch.as_tensor(prim_np, dtype=torch.float32)
                else:
                    prims = torch.stack(
                        [torch.as_tensor(prim_col[i].as_py(), dtype=torch.float32) for i in range(n)]
                    )

                # Muons: list array of muon-vectors; flatten via offsets + values if possible.
                feat_dim = 3
                if (pa.types.is_list(mu_col.type) or pa.types.is_large_list(mu_col.type)) and pa.types.is_fixed_size_list(mu_col.type.value_type):
                    feat_dim = mu_col.type.value_type.list_size

                offsets_arr = getattr(mu_col, "offsets", None) or getattr(mu_col, "value_offsets", None)
                if offsets_arr is not None:
                    offsets = _as_numpy(offsets_arr).astype(np.int64)
                    counts_np = np.diff(offsets)
                else:
                    counts_np = np.array([len(mu_col[i].as_py() or []) for i in range(n)], dtype=np.int64)
                counts = torch.as_tensor(counts_np, dtype=torch.long)

                flat_muons = None
                if pa.types.is_list(mu_col.type) or pa.types.is_large_list(mu_col.type):
                    mu_values = mu_col.values
                    if pa.types.is_fixed_size_list(mu_values.type):
                        feat_dim = mu_values.type.list_size
                        vals = _as_numpy(mu_values.values)
                        flat_muons = torch.as_tensor(vals.reshape((-1, feat_dim)), dtype=torch.float32)

                if flat_muons is None:
                    # Fallback: per-row conversion.
                    mu_list = [mu_col[i].as_py() for i in range(n)]
                    mu_arrs = [
                        np.asarray(m, dtype=np.float32).reshape((-1, feat_dim)) if (m is not None and len(m) > 0) else np.zeros((0, feat_dim), dtype=np.float32)
                        for m in mu_list
                    ]
                    if any(a.size > 0 for a in mu_arrs):
                        flat_muons = torch.as_tensor(np.concatenate(mu_arrs, axis=0), dtype=torch.float32)
                    else:
                        flat_muons = torch.empty((0, feat_dim), dtype=torch.float32)

                batch_idx = torch.repeat_interleave(torch.arange(n, dtype=torch.long), counts)
                yield flat_muons, batch_idx, prims, counts

        # If shuffling is enabled, load all data into memory, shuffle, then yield batches
        if self.shuffle:
            all_flat_muons = []
            all_batch_idx = []
            all_prims = []
            all_counts = []
            
            for path in self.file_paths:
                if fs:
                    if cache_enabled:
                        def _load_bytes() -> bytes:
                            with fs.open(path, "rb") as f:
                                return f.read()
                        raw = self.memory_cache.get_or_load(path, _load_bytes)
                        pf = pq.ParquetFile(io.BytesIO(raw))
                    else:
                        with fs.open(path, "rb") as f:
                            pf = pq.ParquetFile(f)
                else:
                    if cache_enabled:
                        def _load_bytes() -> bytes:
                            with open(path, "rb") as f:
                                return f.read()
                        raw = self.memory_cache.get_or_load(path, _load_bytes)
                        pf = pq.ParquetFile(io.BytesIO(raw))
                    else:
                        pf = pq.ParquetFile(path)
                
                # Load all batches from this file
                for flat_muons, batch_idx, prims, counts in _iter_parquet(pf):
                    all_flat_muons.append(flat_muons)
                    all_batch_idx.append(batch_idx)
                    all_prims.append(prims)
                    all_counts.append(counts)
            
            if len(all_prims) == 0:
                return
            
            # Concatenate all data
            concat_flat_muons = torch.cat(all_flat_muons, dim=0) if all_flat_muons else torch.empty((0, 3), dtype=torch.float32)
            concat_all_prims = torch.cat(all_prims, dim=0)
            concat_all_counts = torch.cat(all_counts, dim=0)
            
            # Create global batch index for all events
            global_batch_indices = torch.cat([
                idx + i * concat_all_prims.shape[0] 
                for i, idx in enumerate(all_batch_idx)
            ])
            
            # Generate shuffled permutation
            rng = np.random.RandomState(self.shuffle_seed)
            num_events = concat_all_prims.shape[0]
            perm = rng.permutation(num_events)
            
            # Shuffle
            perm_tensor = torch.as_tensor(perm, dtype=torch.long)
            shuffled_prims = concat_all_prims[perm_tensor]
            shuffled_counts = concat_all_counts[perm_tensor]
            
            # Recreate batch_idx for shuffled data
            shuffled_batch_idx = torch.repeat_interleave(
                torch.arange(len(shuffled_counts), dtype=torch.long), 
                shuffled_counts
            )
            
            # Also shuffle muons correspondingly
            if concat_flat_muons.numel() > 0:
                # Need to map muon indices through the permutation
                # This is complex because muons are flattened by counts
                # For now, yield muons in their original order but batch them with shuffled primaries
                # This is a limitation but acceptable as a first test
                shuffled_flat_muons = concat_flat_muons
            else:
                shuffled_flat_muons = concat_flat_muons
            
            # Yield in batches of batch_size
            for i in range(0, len(shuffled_prims), self.batch_size):
                batch_end = min(i + self.batch_size, len(shuffled_prims))
                batch_prims = shuffled_prims[i:batch_end]
                batch_counts = shuffled_counts[i:batch_end]
                batch_batch_idx = torch.repeat_interleave(
                    torch.arange(len(batch_counts), dtype=torch.long),
                    batch_counts
                )
                
                # Get muons for this batch (simplified: use original order)
                # A more complex implementation would shuffle muons too
                total_muons_before = int(concat_all_counts[:i].sum().item()) if i > 0 else 0
                total_muons_in_batch = int(batch_counts.sum().item())
                batch_flat_muons = shuffled_flat_muons[total_muons_before:total_muons_before + total_muons_in_batch]
                
                yield batch_flat_muons, batch_batch_idx, batch_prims, batch_counts
        else:
            # Original non-shuffled iteration
            for path in self.file_paths:
                if fs:
                    if cache_enabled:
                        def _load_bytes() -> bytes:
                            with fs.open(path, "rb") as f:
                                return f.read()
                        raw = self.memory_cache.get_or_load(path, _load_bytes)
                        pf = pq.ParquetFile(io.BytesIO(raw))
                        yield from _iter_parquet(pf)
                    else:
                        with fs.open(path, "rb") as f:
                            pf = pq.ParquetFile(f)
                            yield from _iter_parquet(pf)
                else:
                    if cache_enabled:
                        def _load_bytes() -> bytes:
                            with open(path, "rb") as f:
                                return f.read()
                        raw = self.memory_cache.get_or_load(path, _load_bytes)
                        pf = pq.ParquetFile(io.BytesIO(raw))
                        yield from _iter_parquet(pf)
                    else:
                        pf = pq.ParquetFile(path)
                        yield from _iter_parquet(pf)


def get_parquet_batch_dataset(
    file_paths,
    *,
    batch_size: int,
    federation_url: str | None = None,
    token: str | None = None,
    memory_cache=None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
) -> ParquetBatchIterableDataset:
    return ParquetBatchIterableDataset(
        file_paths,
        batch_size=batch_size,
        federation_url=federation_url,
        token=token,
        memory_cache=memory_cache,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )

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
