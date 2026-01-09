"""GPU memory monitoring and tracking utilities for PyTorch training."""

import threading
import time

import torch


def cuda_mem_stats(device: torch.device) -> dict[str, float] | None:
    """Return CUDA memory stats in MiB for logging.

    Note: PyTorch's allocator caches memory; reserved may stay high even when
    allocated drops. Rising *allocated* over time is more indicative of a leak.
    """
    try:
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        idx = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        alloc = float(torch.cuda.memory_allocated(idx)) / (1024.0 * 1024.0)
        reserv = float(torch.cuda.memory_reserved(idx)) / (1024.0 * 1024.0)
        max_alloc = float(torch.cuda.max_memory_allocated(idx)) / (1024.0 * 1024.0)
        max_reserv = float(torch.cuda.max_memory_reserved(idx)) / (1024.0 * 1024.0)

        # allocator internals that help distinguish caching/fragmentation vs a true leak
        active = None
        inactive_split = None
        try:
            stats = torch.cuda.memory_stats(idx)
            active = float(stats.get("active_bytes.all.current", 0.0)) / (1024.0 * 1024.0)
            inactive_split = float(stats.get("inactive_split_bytes.all.current", 0.0)) / (1024.0 * 1024.0)
        except Exception:
            pass

        free_mib = None
        total_mib = None
        try:
            free_b, total_b = torch.cuda.mem_get_info(idx)
            free_mib = float(free_b) / (1024.0 * 1024.0)
            total_mib = float(total_b) / (1024.0 * 1024.0)
        except Exception:
            pass
        return {
            "device_idx": float(idx),
            "alloc_mib": alloc,
            "reserved_mib": reserv,
            "max_alloc_mib": max_alloc,
            "max_reserved_mib": max_reserv,
            "active_mib": float(active) if active is not None else float("nan"),
            "inactive_split_mib": float(inactive_split) if inactive_split is not None else float("nan"),
            "free_mib": float(free_mib) if free_mib is not None else float("nan"),
            "total_mib": float(total_mib) if total_mib is not None else float("nan"),
        }
    except Exception:
        return None


class GPUUsageTracker:
    """Background thread that continuously samples GPU usage for time-averaged reporting."""
    
    def __init__(self, device: torch.device, sample_interval: float = 0.1, window_size: int = 600):
        """
        Args:
            device: torch device to monitor
            sample_interval: seconds between samples (default: 0.1s)
            window_size: max number of samples to keep (default: 600 = 60s at 0.1s intervals)
        """
        self.device = device
        self.sample_interval = float(sample_interval)
        self.window_size = int(window_size)
        self._samples: list[dict[str, float]] = []
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="gpu-usage-tracker", daemon=True)
        
    def start(self) -> None:
        """Start the background sampling thread."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            self._thread.start()
    
    def stop(self) -> None:
        """Stop the background sampling thread."""
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=2)
    
    def _run(self) -> None:
        """Background thread: sample GPU stats at regular intervals."""
        while not self._stop:
            stats = cuda_mem_stats(self.device)
            if stats is not None:
                with self._lock:
                    self._samples.append(stats)
                    # Keep only the most recent window_size samples
                    if len(self._samples) > self.window_size:
                        self._samples.pop(0)
            time.sleep(self.sample_interval)
    
    def get_averaged_stats(self) -> dict[str, float] | None:
        """Return time-averaged GPU stats over the recent sample window."""
        with self._lock:
            if not self._samples:
                return None
            
            # Average all numeric fields across samples
            avg_stats: dict[str, float] = {}
            keys = self._samples[0].keys()
            
            for key in keys:
                values = [s[key] for s in self._samples if key in s]
                if not values:
                    avg_stats[key] = float("nan")
                    continue
                
                # Filter out NaN values for averaging
                valid_values = [v for v in values if v == v]  # NaN != NaN
                if valid_values:
                    avg_stats[key] = sum(valid_values) / len(valid_values)
                else:
                    avg_stats[key] = float("nan")
            
            return avg_stats
