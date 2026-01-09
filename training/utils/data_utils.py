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
