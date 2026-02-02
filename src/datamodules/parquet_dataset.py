import torch
import numpy as np
import io
from collections import OrderedDict

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

import pyarrow as pa
from torch.utils.data import IterableDataset as TorchIterableDataset


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
        multi_file_shuffle: int = 0,
        prefetcher=None,
        original_uris=None,
    ) -> None:
        super().__init__()
        self.file_paths = [str(p) for p in (file_paths or [])]
        self.batch_size = int(batch_size)
        self.federation_url = federation_url
        self.token = token
        self.memory_cache = memory_cache
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)
        self.multi_file_shuffle = int(multi_file_shuffle)
        # Don't store prefetcher - it can't be pickled for multiprocessing
        # Instead, just handle missing files gracefully

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
            concat_all_prims = torch.cat(all_prims, dim=0)
            concat_all_counts = torch.cat(all_counts, dim=0)
            
            # Split flat_muons into individual event tensors to keep association
            if all_flat_muons:
                concat_flat_muons = torch.cat(all_flat_muons, dim=0)
                muon_list = torch.split(concat_flat_muons, concat_all_counts.tolist())
            else:
                muon_list = [torch.empty((0, 3)) for _ in range(concat_all_prims.shape[0])]
            
            # Generate shuffled permutation
            rng = np.random.RandomState(self.shuffle_seed)
            num_events = concat_all_prims.shape[0]
            perm = rng.permutation(num_events)
            
            perm_tensor = torch.as_tensor(perm, dtype=torch.long)
            shuffled_prims = concat_all_prims[perm_tensor]
            shuffled_counts = concat_all_counts[perm_tensor]
            shuffled_muon_list = [muon_list[i] for i in perm]
            
            # Yield in batches of batch_size
            for i in range(0, num_events, self.batch_size):
                batch_end = min(i + self.batch_size, num_events)
                batch_prims = shuffled_prims[i:batch_end]
                batch_counts = shuffled_counts[i:batch_end]
                
                # Combine muons for this batch
                batch_muons = shuffled_muon_list[i:batch_end]
                batch_flat_muons = torch.cat(batch_muons, dim=0) if any(m.numel() > 0 for m in batch_muons) else torch.empty((0, 3))
                
                batch_batch_idx = torch.repeat_interleave(
                    torch.arange(len(batch_counts), dtype=torch.long),
                    batch_counts
                )
                
                yield batch_flat_muons, batch_batch_idx, batch_prims, batch_counts
        elif self.multi_file_shuffle > 0:
            # Interleave batches from multiple files concurrently
            num_concurrent = min(self.multi_file_shuffle, len(self.file_paths))
            
            # Shuffle initial file list to randomize which files start together
            rng = np.random.RandomState(self.shuffle_seed)
            shuffled_paths = list(self.file_paths)
            rng.shuffle(shuffled_paths)
            
            # Use a pool of iterators
            iterators = []
            
            def _get_pf(path):
                if fs:
                    if cache_enabled:
                        def _load_bytes() -> bytes:
                            with fs.open(path, "rb") as f:
                                return f.read()
                        raw = self.memory_cache.get_or_load(path, _load_bytes)
                        return pq.ParquetFile(io.BytesIO(raw))
                    else:
                        return pq.ParquetFile(fs.open(path, "rb"))
                else:
                    if cache_enabled:
                        def _load_bytes() -> bytes:
                            with open(path, "rb") as f:
                                return f.read()
                        raw = self.memory_cache.get_or_load(path, _load_bytes)
                        return pq.ParquetFile(io.BytesIO(raw))
                    else:
                        return pq.ParquetFile(path)

            path_queue = list(shuffled_paths)
            
            # Initialize the pool
            while len(iterators) < num_concurrent and path_queue:
                p = path_queue.pop(0)
                try:
                    pf = _get_pf(p)
                    iterators.append(_iter_parquet(pf))
                except Exception as e:
                    print(f"Error opening {p}: {e}")

            while iterators:
                # Pick a random iterator from the active pool
                idx = rng.randint(0, len(iterators))
                try:
                    yield next(iterators[idx])
                except StopIteration:
                    # File finished, replace with next from queue if available
                    iterators.pop(idx)
                    if path_queue:
                        p = path_queue.pop(0)
                        try:
                            pf = _get_pf(p)
                            iterators.append(_iter_parquet(pf))
                        except Exception as e:
                            print(f"Error opening {p}: {e}")
        else:
            # Original non-shuffled iteration
            import time
            import os
            
            for path in self.file_paths:
                # Handle missing files gracefully (they may still be downloading via prefetcher)
                max_wait_time = 60  # Max 60 seconds per file
                wait_interval = 0.5
                waited = 0
                
                # Wait for file to exist (if using local paths)
                if not fs:  # Local filesystem
                    while not os.path.exists(path) and waited < max_wait_time:
                        time.sleep(wait_interval)
                        waited += wait_interval
                    
                    if not os.path.exists(path):
                        # File still doesn't exist after waiting, skip it
                        continue
                
                try:
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
                except FileNotFoundError:
                    # File not yet available (still being prefetched), skip for now
                    continue
                except Exception as e:
                    # Log other errors but continue with remaining files
                    print(f"Error processing {path}: {e}")
                    continue


def get_parquet_batch_dataset(
    file_paths,
    *,
    batch_size: int,
    federation_url: str | None = None,
    token: str | None = None,
    memory_cache=None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    multi_file_shuffle: int = 0,
    prefetcher=None,
    original_uris=None,
) -> ParquetBatchIterableDataset:
    return ParquetBatchIterableDataset(
        file_paths,
        batch_size=batch_size,
        federation_url=federation_url,
        token=token,
        memory_cache=memory_cache,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        multi_file_shuffle=multi_file_shuffle,
        prefetcher=prefetcher,
        original_uris=original_uris,
    )
