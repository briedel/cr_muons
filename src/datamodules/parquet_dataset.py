import torch
import numpy as np
import io
import os
import time
from collections import OrderedDict

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

import pyarrow as pa
from torch.utils.data import IterableDataset as TorchIterableDataset


def _rebatch_stream(source_iter, batch_size):
    """Ensure all batches yielded from the source_iter have exactly batch_size
    primaries, except possibly the very last batch of the stream.
    """
    rem_p, rem_m, rem_c = None, None, None
    rng = np.random.RandomState(42) # Fixed seed for reproducible batch shuffling
    
    for m, _, p, c in source_iter:
        if p.size(0) == 0:
            continue
            
        if rem_p is not None:
            p = torch.cat([rem_p, p], dim=0)
            m = torch.cat([rem_m, m], dim=0)
            c = torch.cat([rem_c, c], dim=0)
            rem_p, rem_m, rem_c = None, None, None
        
        n_total = p.size(0)
        curr_m_start = 0
        for start in range(0, n_total, batch_size):
            end = start + batch_size
            if end <= n_total:
                p_batch = p[start:end]
                c_batch = c[start:end]
                m_n = int(c_batch.sum())
                m_batch = m[curr_m_start : curr_m_start + m_n]
                curr_m_start += m_n
                
                # Shuffle the batch explicitly before yielding
                # We shuffle primaries indices, then apply to primaries and counts.
                # Muons must be regrouped or we just rely on PyTorch Geometric style logic later?
                # The code below assumes (muons, idx, prims, counts) structure.
                # To shuffle correctly, we need to permute prims/counts, and re-arrange muons accordingly.
                
                perm = rng.permutation(batch_size)
                perm_t = torch.as_tensor(perm, dtype=torch.long)
                
                p_batch = p_batch[perm_t]
                c_batch = c_batch[perm_t]
                
                # Re-constructing the flat muon tensor for the shuffled events is expensive
                # (requires splitting m_batch by c_batch then re-stacking).
                # A faster way: if we don't strictly need muons to be physically sequential in memory
                # relative to primaries in specific order, we could just pass them.
                # BUT: The downstream model likely expects muons[idx_batch == i] to correspond to prims[i].
                # So we MUST keep limits aligned.
                
                # Since splitting is slow, let's process the split only if we shuffled.
                # Actually, doing the split is the only robust way to shuffle variable-length data.
                if m_n > 0:
                   # Fast split via CPU list comprehension usually beats pure torch for simple ragged structures 
                   # if counts are small integers.
                   # But let's use torch.split which is standard. 
                   mu_splits = torch.split(m_batch, c_batch.tolist())
                   # Reorder list
                   mu_splits = [mu_splits[i] for i in perm]
                   m_batch = torch.cat(mu_splits, dim=0)
                
                # Re-generate clean batch indices
                idx_batch = torch.repeat_interleave(
                    torch.arange(batch_size, dtype=torch.long, device=p_batch.device),
                    c_batch
                )
                yield m_batch, idx_batch, p_batch, c_batch
            else:
                rem_p = p[start:]
                rem_m = m[curr_m_start:]
                rem_c = c[start:]
                # if rem_p.size(0) > (batch_size // 2) and batch_size > 1000:
                #    print(f"  [Rebatcher] Buffered {rem_p.size(0)}/{batch_size} primaries...")
                
    if rem_p is not None:
        idx_batch = torch.repeat_interleave(
            torch.arange(rem_p.size(0), dtype=torch.long, device=rem_p.device),
            rem_c
        )
        yield rem_m, idx_batch, rem_p, rem_c


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
        drop_empty_events: bool = False,
        prefetcher=None,
        original_uris=None,
        delete_after_use: bool = False,
        processed_files_shared=None,
        limit_files_per_epoch: int = 0,
        muon_feature_selection: str = "all",
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
        self.drop_empty_events = bool(drop_empty_events)
        self.original_uris = original_uris
        self.delete_after_use = delete_after_use
        self.processed_files_shared = processed_files_shared
        self.limit_files_per_epoch = limit_files_per_epoch
        self.muon_feature_selection = muon_feature_selection
        
        # We can't pickle the whole prefetcher (it contains a thread),
        # but we can pickle the shared index and URI-to-index map.
        self._current_index_shared = getattr(prefetcher, "_current_index_shared", None)
        self.uri_to_index = getattr(prefetcher, "uri_to_index", None)

    def __iter__(self):
        if pq is None:
            raise ImportError("pyarrow is required to read parquet files. Install with `pip install pyarrow`")

        import pyarrow as pa
        import glob

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

        # Filter out already processed files BEFORE sharding
        all_files_to_process = list(self.file_paths)
        if self.processed_files_shared is not None:
            processed_set = set(self.processed_files_shared)
            
            # Helper to check if a local path corresponds to a processed original URI
            local_to_orig = {}
            if self.original_uris:
                for lp, ou in zip(self.file_paths, self.original_uris):
                    local_to_orig[lp] = ou
            
            initial_count = len(all_files_to_process)
            all_files_to_process = [
                f for f in all_files_to_process 
                if f not in processed_set and local_to_orig.get(f) not in processed_set
            ]
            skipped = initial_count - len(all_files_to_process)
            if skipped > 0:
                print(f"[Worker {shard_id}] Skipping {skipped}/{initial_count} already processed files.")

        if self.shuffle:
            # Shuffle the remaining list before sharding
            # BUT only if we aren't using a prefetcher (which relies on sequential order matches)
            if self._current_index_shared is None:
                rng_files = np.random.RandomState(self.shuffle_seed)
                rng_files.shuffle(all_files_to_process)
        
        file_paths = [
            f for i, f in enumerate(all_files_to_process)
            if (i % shard_world) == shard_id
        ]
        
        if not file_paths:
            return

        # Limit files per epoch (per worker)
        files_finished_this_epoch = 0
        worker_limit = 0
        if self.limit_files_per_epoch > 0:
            # Each worker gets a proportional share of the limit
            worker_limit = max(1, self.limit_files_per_epoch // shard_world)
            if (shard_id < (self.limit_files_per_epoch % shard_world)):
                worker_limit += 1
        
        # Explicitly report the files this worker is taking
        names = [os.path.basename(f) for f in file_paths]
        if len(names) > 8:
            names_summary = ", ".join(names[:4]) + " ... " + ", ".join(names[-4:])
            print(f"[Worker {shard_id}] using {len(names)} files: {names_summary}", flush=True)
        else:
            print(f"[Worker {shard_id}] using these files: {', '.join(names)}", flush=True)

        fs = None
        if self.federation_url:
            try:
                from pelicanfs.core import PelicanFileSystem
            except ImportError:
                raise ImportError("pelicanfs is required. Install with: pip install pelicanfs")

            headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
            fs = PelicanFileSystem(self.federation_url, headers=headers)

        cache_enabled = bool(self.memory_cache is not None and getattr(self.memory_cache, "enabled", lambda: False)())

        # Map local paths to indices for prefetcher progress updates
        path_to_index = {}
        path_to_orig_uri = {}
        if self.original_uris:
            # Important: map based on the UN-SHARDED indices from the original URI list
            # to ensure we accurately report progress to the downloader.
            for i, local_p in enumerate(self.file_paths):
                if i < len(self.original_uris):
                    orig_uri = self.original_uris[i]
                    path_to_orig_uri[local_p] = orig_uri
                    if self.uri_to_index:
                        idx = self.uri_to_index.get(orig_uri)
                        if idx is not None:
                            path_to_index[local_p] = idx

        def _as_numpy(a) -> np.ndarray:
            return a.to_numpy(zero_copy_only=False)

        def _get_pf(path, timeout=600):
            # Helper to open a ParquetFile with wait logic for local files
            import time
            import os
            
            # Update prefetcher progress BEFORE opening/waiting
            # This allows the background thread to start downloading immediately
            if self._current_index_shared is not None:
                idx = path_to_index.get(path)
                if idx is not None:
                    with self._current_index_shared.get_lock():
                        if idx > self._current_index_shared.value:
                            self._current_index_shared.value = idx

            if fs:
                if cache_enabled:
                    def _load_bytes() -> bytes:
                        with fs.open(path, "rb") as f:
                            return f.read()
                    raw = self.memory_cache.get_or_load(path, _load_bytes)
                    pf = pq.ParquetFile(io.BytesIO(raw))
                else:
                    pf = pq.ParquetFile(fs.open(path, "rb"))
            else:
                # Handle missing local files (e.g. still downloading via prefetcher)
                max_wait_time = timeout
                wait_interval = 0.5
                waited = 0
                while not os.path.exists(path) and waited < max_wait_time:
                    time.sleep(wait_interval)
                    waited += wait_interval
                
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found after {max_wait_time}s: {path}")

                if cache_enabled:
                    def _load_bytes() -> bytes:
                        with open(path, "rb") as f:
                            return f.read()
                    raw = self.memory_cache.get_or_load(path, _load_bytes)
                    pf = pq.ParquetFile(io.BytesIO(raw))
                else:
                    pf = pq.ParquetFile(path)
                
                # Verify file content for the user
                n_rows = pf.metadata.num_rows
                n_groups = pf.metadata.num_row_groups
                print(f"  + File {os.path.basename(path)} has {n_rows} total rows ({n_groups} row groups)")

            return pf

        def _mark_done(path):
            """Mark a file as processed and optionally delete it."""
            nonlocal files_finished_this_epoch
            files_finished_this_epoch += 1

            # Determine the name/URI to record
            record_path = path_to_orig_uri.get(path, path)

            if self.processed_files_shared is not None:
                self.processed_files_shared.append(record_path)
            
            basename = os.path.basename(path)
            msg = f"[Worker {shard_id}] done with {basename}"
            
            if self.delete_after_use and not fs:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        # Also attempt to remove .tmp files if any (though they shouldn't be here)
                        tmp_files = glob.glob(path + ".*.tmp")
                        for tmp in tmp_files:
                            os.remove(tmp)
                        msg = f"[Worker {shard_id}] done with {basename} and deleting local copy"
                except Exception as e:
                    msg += f" (error deleting: {e})"
            
            print(msg, flush=True)

        def _iter_parquet(pf, path_for_logging=None):
            cols = ["primary", "muons"]
            # Cap internal chunk size to avoid massive Python-loop overhead.
            # 128k is a good balance for IO vs processing speed.
            internal_rb_size = min(131072, max(4096, self.batch_size)) 
            
            total_yielded = 0
            total_found = 0

            # Progress tracking
            total_rows_in_file = pf.metadata.num_rows
            processed_rows_in_file = 0
            last_log_threshold = 0
            basename = os.path.basename(path_for_logging) if path_for_logging else "unknown"
            
            for rb in pf.iter_batches(batch_size=internal_rb_size, columns=cols):
                n_rb = rb.num_rows
                processed_rows_in_file += n_rb

                # Log progress every 20% to avoid log spam but give confidence
                progress = processed_rows_in_file / total_rows_in_file if total_rows_in_file > 0 else 0
                if progress - last_log_threshold >= 0.05:
                    print(f"[Worker {shard_id}] {basename}: {progress*100:.1f}% ({processed_rows_in_file}/{total_rows_in_file})", flush=True)
                    last_log_threshold = progress

                if n_rb <= 0:
                    continue
                total_found += n_rb

                # 1. Convert whole record batch to Tensors
                prim_col = rb.column(0)
                mu_col = rb.column(1)

                # Primaries: try a vectorized path for both fixed and variable lists.
                if pa.types.is_list(prim_col.type) or pa.types.is_large_list(prim_col.type) or pa.types.is_fixed_size_list(prim_col.type):
                    try:
                        p_vals = _as_numpy(prim_col.values)
                        p_dim = prim_col.type.list_size if pa.types.is_fixed_size_list(prim_col.type) else (len(p_vals) // n_rb)
                        # Use torch.tensor() to copy into writable memory and avoid UserWarning
                        prims = torch.tensor(p_vals.reshape((n_rb, p_dim)), dtype=torch.float32)
                    except Exception:
                        # Fallback for truly ragged primaries (unlikely for physics data)
                        prims = torch.stack(
                            [torch.as_tensor(prim_col[i].as_py(), dtype=torch.float32) for i in range(n_rb)]
                        )
                else:
                    prims = torch.stack(
                        [torch.as_tensor(prim_col[i].as_py(), dtype=torch.float32) for i in range(n_rb)]
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
                    counts_np = np.array([len(mu_col[i].as_py() or []) for i in range(n_rb)], dtype=np.int64)
                counts = torch.as_tensor(counts_np, dtype=torch.long)

                flat_muons = None
                if pa.types.is_list(mu_col.type) or pa.types.is_large_list(mu_col.type):
                    mu_values = mu_col.values
                    if pa.types.is_fixed_size_list(mu_values.type):
                        feat_dim = mu_values.type.list_size
                        vals = _as_numpy(mu_values.values)
                        # Use torch.tensor() to copy into writable memory and avoid UserWarning
                        flat_muons = torch.tensor(vals.reshape((-1, feat_dim)), dtype=torch.float32)

                if flat_muons is None:
                    # Fallback: per-row conversion.
                    mu_list = [mu_col[i].as_py() for i in range(n_rb)]
                    
                    # Dynamically detect feature dimension from the first non-empty event
                    # This allows supporting both 3-feature (E,x,y) and 4-feature (E,r,x,y) formats
                    running_feat_dim = feat_dim
                    for m in mu_list:
                        if m and len(m) > 0:
                            # m is a list of lists [[f1, f2...], [f1, f2...]]
                            if len(m) > 0 and isinstance(m[0], (list, tuple)):
                                running_feat_dim = len(m[0])
                                break
                    
                    if running_feat_dim != feat_dim:
                        # Update default if we found a different dimension
                        feat_dim = running_feat_dim

                    mu_arrs = [
                        np.asarray(m, dtype=np.float32).reshape((-1, feat_dim)) if (m is not None and len(m) > 0) else np.zeros((0, feat_dim), dtype=np.float32)
                        for m in mu_list
                    ]
                    if any(a.size > 0 for a in mu_arrs):
                        flat_muons = torch.tensor(np.concatenate(mu_arrs, axis=0), dtype=torch.float32)
                    else:
                        flat_muons = torch.empty((0, feat_dim), dtype=torch.float32)

                # Feature Selection / Slicing if requested (e.g. use r instead of x,y)
                # Assumes source is [E, r, x, y] (4 features) or [E, x, y] (3 features)
                if flat_muons.size(0) > 0 and self.muon_feature_selection != "all":
                    # If we only have 3 features (E, x, y), we assume they are indices 0, 1, 2
                    # If we have 4 features (E, r, x, y), indices are 0, 1, 2, 3
                    
                    if feat_dim == 4:
                        if self.muon_feature_selection == "xy":
                            # Use E, x, y -> indices 0, 2, 3
                            flat_muons = flat_muons[:, [0, 2, 3]]
                        elif self.muon_feature_selection == "r":
                             # Use E, r -> indices 0, 1
                             # NOTE: This reduces feat_dim to 2. Model must expect it.
                             # Or maybe users wants E, r, z? Usually 3D is needed.
                             # But user asked: "uses r OR x,y".
                             # Assuming they mean using (Energy, Radius) vs (Energy, X, Y).
                             flat_muons = flat_muons[:, [0, 1]]
                    elif feat_dim == 3 and self.muon_feature_selection == "r":
                        raise ValueError("Cannot select 'r' feature if only 3 features are present. Expected format with 'r' is [E, r, x, y].")
                    elif feat_dim == 3 and self.muon_feature_selection == "xy":
                        # Already x,y. Can't select "r" unless we compute it or it was there.
                        # Assuming 3-col files are E, x, y.
                        pass


                # 2. Optional: Drop empty events
                if self.drop_empty_events:
                    keep_mask = counts > 0
                    if not keep_mask.any():
                        continue
                    
                    prims = prims[keep_mask]
                    # Filter muons (use original counts to find them)
                    full_batch_idx = torch.repeat_interleave(torch.arange(n_rb, dtype=torch.long), counts)
                    muon_keep_mask = keep_mask[full_batch_idx]
                    flat_muons = flat_muons[muon_keep_mask]
                    counts = counts[keep_mask]

                total_yielded += counts.size(0)
                # Yield unaligned record batch. batch_idx is None as it will be handled by rebatcher.
                yield flat_muons, None, prims, counts
            
            if path_for_logging:
                fname = os.path.basename(path_for_logging)
                print(f"  + Finished {fname}: {total_yielded}/{total_found} events had muons.")


        # Use a generator to encapsulate all traversal paths, then wrap with alignment logic.
        def _get_raw_stream():
            # If shuffling is enabled, load data into memory in chunks (defined by multi_file_shuffle), 
            # shuffle the chunk, then yield batches. This allows "buffer-and-shuffle" execution.
            if self.shuffle:
                files_to_process = list(file_paths)
                # If multi_file_shuffle is set, use it as chunk size ("N files").
                # If 0 (default), load ALL files (infinite chunk).
                chunk_size = self.multi_file_shuffle if self.multi_file_shuffle > 0 else len(files_to_process)
                
                rng = np.random.RandomState(self.shuffle_seed)

                for i in range(0, len(files_to_process), chunk_size):
                    chunk_files = files_to_process[i : i + chunk_size]
                    
                    if shard_world > 1:
                        # Only print simple status for chunks to avoid log spam, strict logging in _mark_done handles the details
                        pass

                    all_flat_muons = []
                    all_prims = []
                    all_counts = []
                    
                    for path in chunk_files:
                        if worker_limit > 0 and files_finished_this_epoch >= worker_limit:
                            break

                        try:
                            # Wait for file (using timeout)
                            pf = _get_pf(path, timeout=600)
                        except (FileNotFoundError, Exception) as e:
                            print(f"Error opening {path}: {e}")
                            continue
                        
                        # Load all batches from this file
                        for flat_muons, _, prims, counts in _iter_parquet(pf, path_for_logging=path):
                            all_flat_muons.append(flat_muons)
                            all_prims.append(prims)
                            all_counts.append(counts)                
                        _mark_done(path)            
                    
                    if len(all_prims) == 0:
                        continue
                    
                    # Concatenate all data
                    concat_all_prims = torch.cat(all_prims, dim=0)
                    concat_all_counts = torch.cat(all_counts, dim=0)
                    
                    # Split flat_muons into individual event tensors to keep association
                    if all_flat_muons:
                        concat_flat_muons = torch.cat(all_flat_muons, dim=0)
                        muon_list = torch.split(concat_flat_muons, concat_all_counts.tolist())
                    else:
                        muon_list = [torch.empty((0, 3)) for _ in range(concat_all_prims.shape[0])]
                    
                    # Generate shuffled permutation for this chunk
                    num_events = concat_all_prims.shape[0]
                    perm = rng.permutation(num_events)
                    
                    perm_tensor = torch.as_tensor(perm, dtype=torch.long)
                    shuffled_prims = concat_all_prims[perm_tensor]
                    shuffled_counts = concat_all_counts[perm_tensor]
                    shuffled_muon_list = [muon_list[i] for i in perm]
                    
                    # Yield individually so the top-level rebatcher can regroup them properly.
                    for j in range(num_events):
                        m_ev = shuffled_muon_list[j]
                        p_ev = shuffled_prims[j:j+1]
                        c_ev = shuffled_counts[j:j+1]
                        yield m_ev, None, p_ev, c_ev

            elif self.multi_file_shuffle > 0:
                # Interleave batches from multiple files concurrently
                num_concurrent = min(self.multi_file_shuffle, len(file_paths))
                
                # Shuffle initial file list to randomize which files start together
                rng = np.random.RandomState(self.shuffle_seed)
                shuffled_paths = list(file_paths)
                
                # IMPORTANT: If using a prefetcher (indicated by shared index), we MUST preserve Sequential Order
                # because the prefetcher triggers downloads based on max(index).
                # Random shuffling causes the index to jump ahead, triggering massive unnecessary downloads.
                if self._current_index_shared is None:
                    rng.shuffle(shuffled_paths)
                
                # Use a pool of iterators
                iterators = []
                path_queue = list(shuffled_paths)
                
                if shard_world > 1:
                    print(f"[Worker {shard_id}] Starting shuffle with {len(file_paths)} files (pool size {num_concurrent})")
                else:
                    print(f"Starting shuffle with {len(file_paths)} files (pool size {num_concurrent})")

                # Initialize the pool
                searched = 0
                max_to_search = len(path_queue)
                while len(iterators) < num_concurrent and path_queue and searched < max_to_search:
                    p = path_queue.pop(0)
                    searched += 1
                    if fs or os.path.exists(p):
                        try:
                            pf = _get_pf(p, timeout=0) 
                            iterators.append((_iter_parquet(pf, path_for_logging=p), p))
                        except Exception as e:
                            print(f"Error opening ready file {p}: {e}")
                    else:
                        path_queue.append(p)
                        if not iterators:
                            try:
                                pf = _get_pf(p, timeout=30)
                                iterators.append((_iter_parquet(pf, path_for_logging=p), p))
                            except Exception:
                                time.sleep(1)
                
                # Report initial pool state to user
                pool_basenames = [os.path.basename(x[1]) for x in iterators]
                print(f"[Worker {shard_id}] Streaming concurrently from: {pool_basenames}", flush=True)

                chunks_count = 0
                while iterators:
                    chunks_count += 1
                    # Log periodically to show which files are actively being read
                    # 50 chunks is roughly 50 * 128k = 6.4M events, or ~15-20% of the pool's progress
                    if chunks_count % 50 == 0:
                        curr_active = [os.path.basename(x[1]) for x in iterators]
                        print(f"[Worker {shard_id}] Pool Update: still processing {curr_active}", flush=True)


                    idx = rng.randint(0, len(iterators))
                    it, p_current = iterators[idx]
                    try:
                        yield next(it)
                    except StopIteration:
                        iterators.pop(idx)
                        _mark_done(p_current)
                        
                        if worker_limit > 0 and files_finished_this_epoch >= worker_limit:
                            continue

                        # Refill logic: keep trying to add files until we succeed or run out of options
                        # taking care not to exit if we simply had a timeout but have other files.
                        while path_queue and len(iterators) < num_concurrent:
                            p = path_queue.pop(0)
                            try:
                                pf = _get_pf(p, timeout=60)
                                iterators.append((_iter_parquet(pf, path_for_logging=p), p))
                            except FileNotFoundError:
                                # Start logging these as errors/warnings instead of silently re-queueing
                                # confusing the loop exit condition.
                                print(f"[Worker {shard_id}] Timeout waiting for {os.path.basename(p)}. Skipping file to keep worker alive.")
                            except Exception as e:
                                print(f"[Worker {shard_id}] Error refilling pool with {p}: {e}")
            else:
                # Original non-shuffled iteration
                for path in file_paths:
                    if worker_limit > 0 and files_finished_this_epoch >= worker_limit:
                        break

                    try:
                        pf = _get_pf(path, timeout=600)
                        yield from _iter_parquet(pf, path_for_logging=path)
                        _mark_done(path)
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
                        continue
            
            if worker_limit > 0 and files_finished_this_epoch >= worker_limit:
                print(f"[Worker {shard_id}] Reached virtual epoch limit of {files_finished_this_epoch} files.")

        return _rebatch_stream(_get_raw_stream(), self.batch_size)


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
    drop_empty_events: bool = False,
    prefetcher=None,
    original_uris=None,
    delete_after_use: bool = False,
    processed_files_shared=None,
    limit_files_per_epoch: int = 0,
    muon_feature_selection: str = "all",
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
        drop_empty_events=drop_empty_events,
        prefetcher=prefetcher,
        original_uris=original_uris,
        delete_after_use=delete_after_use,
        processed_files_shared=processed_files_shared,
        limit_files_per_epoch=limit_files_per_epoch,
        muon_feature_selection=muon_feature_selection,
    )
