"""Data handling utilities for training."""

import os
import queue
import threading

import torch


class OutlierParquetWriter:
    """Write outlier events to individual Parquet files."""

    def __init__(self, out_dir: str) -> None:
        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self._counter = 0

    def write_event(
        self,
        *,
        source_file: str,
        source_file_index: int,
        batch_index: int,
        event_index: int,
        count: int,
        primaries: object,
        muons: object,
    ) -> str:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as e:
            raise ImportError(
                "Writing outliers requires pyarrow. Install with `pip install pyarrow`."
            ) from e

        def _to_py(x: object):
            if isinstance(x, torch.Tensor):
                return x.detach().to("cpu").tolist()
            return x

        row = {
            "source_file": str(source_file),
            "source_file_index": int(source_file_index),
            "batch_index": int(batch_index),
            "event_index": int(event_index),
            "count": int(count),
            "primaries": _to_py(primaries),
            "muons": _to_py(muons),
        }

        table = pa.Table.from_pylist([row])
        out_path = os.path.join(self.out_dir, f"part-{self._counter:09d}.parquet")
        pq.write_table(table, out_path)
        self._counter += 1
        return out_path


class PrefetchIterator:
    """Background thread that prefetches items from an iterator."""

    def __init__(self, base_iter, *, max_prefetch: int) -> None:
        self._base_iter = base_iter
        self._max_prefetch = max(0, int(max_prefetch))
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=self._max_prefetch or 1)
        self._sentinel = object()
        self._exc: Exception | None = None
        self._thread = None

        if self._max_prefetch > 0:
            self._thread = threading.Thread(target=self._run, name="batch-prefetch", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            for item in self._base_iter:
                self._q.put(item)
            self._q.put(self._sentinel)
        except Exception as e:
            self._exc = e
            self._q.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        if self._max_prefetch <= 0:
            return next(self._base_iter)

        item = self._q.get()
        if item is self._sentinel:
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item


class MultiFileShuffledIterator:
    """Iterate over multiple files simultaneously, yielding batches in round-robin order.
    
    This prevents overfitting to individual file distributions by mixing batches across
    multiple files. When a file is exhausted, it's replaced with the next file from the queue.
    
    Args:
        file_queue: List of file paths to process
        loader_factory: Callable that takes a file path and returns a DataLoader
        num_concurrent: Number of files to keep open simultaneously
    
    Yields:
        Tuple of (batch_data, file_info_dict) where file_info_dict contains:
            - 'file_path': source file path
            - 'file_idx': global file index (1-based)
            - 'batches_from_file': number of batches yielded from this file
            - 'exhausted': whether this file is now exhausted
    """

    def __init__(self, file_queue: list, loader_factory, num_concurrent: int = 10):
        self.file_queue = list(file_queue)  # Make a copy
        self.loader_factory = loader_factory
        self.num_concurrent = num_concurrent
        self.total_files = len(file_queue)
        
        # Track active loaders
        self.active_loaders: list = []  # [(file_path, file_idx, iterator, batches_from_file)]
        self.next_file_idx = 0
        self.round_robin_idx = 0
        self.batches_yielded = 0
        
    def _open_next_file(self):
        """Open the next file from the queue and add it to active loaders."""
        if self.next_file_idx >= self.total_files:
            return False
            
        file_path = self.file_queue[self.next_file_idx]
        file_idx = self.next_file_idx + 1  # 1-based indexing
        
        try:
            loader = self.loader_factory(file_path)
            iterator = iter(loader)
            self.active_loaders.append({
                'file_path': file_path,
                'file_idx': file_idx,
                'iterator': iterator,
                'batches_from_file': 0,
                'loader': loader,  # Keep reference to prevent GC
            })
            self.next_file_idx += 1
            return True
        except Exception as e:
            # Log error but continue with other files
            print(f"Warning: Failed to open file {file_path}: {e}")
            self.next_file_idx += 1
            return self._open_next_file() if self.next_file_idx < self.total_files else False
    
    def __iter__(self):
        # Initialize by opening num_concurrent files
        for _ in range(min(self.num_concurrent, self.total_files)):
            if not self._open_next_file():
                break
        
        return self
    
    def __next__(self):
        if not self.active_loaders:
            raise StopIteration
        
        # Round-robin across active loaders
        attempts = 0
        max_attempts = len(self.active_loaders) * 2  # Allow full round + buffer
        
        while attempts < max_attempts:
            if not self.active_loaders:
                raise StopIteration
            
            # Wrap around if needed
            if self.round_robin_idx >= len(self.active_loaders):
                self.round_robin_idx = 0
            
            loader_info = self.active_loaders[self.round_robin_idx]
            
            try:
                # Try to get next batch from current file
                batch = next(loader_info['iterator'])
                loader_info['batches_from_file'] += 1
                self.batches_yielded += 1
                
                # Prepare file info to return with batch
                file_info = {
                    'file_path': loader_info['file_path'],
                    'file_idx': loader_info['file_idx'],
                    'batches_from_file': loader_info['batches_from_file'],
                    'exhausted': False,
                    'active_files': len(self.active_loaders),
                    'total_files': self.total_files,
                }
                
                # Move to next file for next call
                self.round_robin_idx += 1
                
                return batch, file_info
                
            except StopIteration:
                # This file is exhausted
                exhausted_info = {
                    'file_path': loader_info['file_path'],
                    'file_idx': loader_info['file_idx'],
                    'batches_from_file': loader_info['batches_from_file'],
                    'exhausted': True,
                    'active_files': len(self.active_loaders) - 1,
                    'total_files': self.total_files,
                }
                
                # Remove exhausted loader
                self.active_loaders.pop(self.round_robin_idx)
                
                # Try to replace with next file from queue
                opened = self._open_next_file()
                
                # Adjust round_robin_idx if needed
                if self.round_robin_idx >= len(self.active_loaders):
                    self.round_robin_idx = 0
                
                # Yield a special marker for exhausted file
                # Caller can check file_info['exhausted'] to handle cleanup
                if exhausted_info['batches_from_file'] > 0:
                    # Only yield exhausted marker if we actually processed batches
                    return None, exhausted_info
                else:
                    # Empty file, continue to next
                    attempts += 1
                    continue
            
            attempts += 1
        
        # If we exhausted all attempts, we're done
        raise StopIteration
    
    def get_progress_string(self):
        """Return a string describing current progress."""
        active = len(self.active_loaders)
        completed = self.next_file_idx - active
        return f"{completed}/{self.total_files} files, {active} active"
