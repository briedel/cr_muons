import argparse
import json
import os
import tempfile
import fnmatch
import posixpath
import subprocess
import sys
import shutil
import hashlib
import threading
import time
import queue
from urllib.parse import urlparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None

from pathlib import Path
from dataloader import SingleHDF5Dataset, MultiHDF5Dataset, ragged_collate_fn
from hf_dataloader import FileBytesLRUCache, get_hf_dataset, get_parquet_batch_dataset, hf_collate_fn
from normalizer import DataNormalizer
from model import ScalableGenerator, ScalableCritic, train_step_scalable

from torch.utils.data import DataLoader


def _has_wildcards(s: str) -> bool:
    return any(ch in s for ch in ("*", "?", "["))


def _infer_file_format(file_path: str) -> str:
    p = str(file_path).lower()
    if p.endswith(".parquet") or p.endswith(".pq"):
        return "parquet"
    if p.endswith(".h5") or p.endswith(".hdf5"):
        return "h5"
    # Default to HDF5 for backward compatibility.
    return "h5"


def _infer_pelican_federation_url(pelican_uri: str) -> str | None:
    """Infer federation discovery URL (e.g. pelican://osg-htc.org) from a pelican URI."""
    try:
        u = urlparse(pelican_uri)
    except Exception:
        return None
    if u.scheme != "pelican" or not u.netloc:
        return None
    return f"pelican://{u.netloc}"


def _pelican_uri_to_local_cache_path(pelican_uri: str, *, cache_dir: str) -> str:
    """Map a pelican:// URI to a stable local cache path.

    We preserve the full path under the cache directory to avoid collisions.
    Example:
      pelican://osg-htc.org/icecube/wipac/a/b/file.parquet
      -> <cache_dir>/icecube/wipac/a/b/file.parquet
    """
    u = urlparse(pelican_uri)
    rel = (u.path or "/").lstrip("/")
    return os.path.join(str(cache_dir), rel)


def _prefetch_pelican_files(
    pelican_uris: list[str],
    *,
    fs,
    cache_dir: str,
) -> list[str]:
    """Download pelican:// inputs to a local cache directory.

    Returns the list of local file paths (same order as input URIs).
    """
    cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    local_paths: list[str] = []
    for uri in pelican_uris:
        uri = str(uri)
        out_path = _pelican_uri_to_local_cache_path(uri, cache_dir=cache_dir)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            local_paths.append(out_path)
            continue

        tmp_path = out_path + ".tmp"
        print(f"Prefetching {uri} -> {out_path}")
        with fs.open(uri, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        os.replace(tmp_path, out_path)
        local_paths.append(out_path)

    return local_paths


class _PelicanPrefetcher:
    def __init__(
        self,
        pelican_uris: list[str],
        *,
        federation_url: str,
        token: str | None,
        cache_dir: str,
        ahead: int,
    ) -> None:
        self.pelican_uris = [str(u) for u in pelican_uris]
        self.uri_to_index = {u: i for i, u in enumerate(self.pelican_uris)}
        self.federation_url = federation_url
        self.token = token
        self.cache_dir = str(cache_dir)
        self.ahead = max(0, int(ahead))

        self._current_index = -1
        self._next_download = 0
        self._stop = False
        self._status: dict[str, str] = {u: "pending" for u in self.pelican_uris}
        self._errors: dict[str, str] = {}

        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="pelican-prefetch", daemon=True)

    def start(self) -> None:
        if self.pelican_uris:
            self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def update_current_uri(self, uri: str) -> None:
        idx = self.uri_to_index.get(str(uri))
        if idx is None:
            return
        with self._cond:
            if idx > self._current_index:
                self._current_index = idx
                self._cond.notify_all()

    def local_path(self, uri: str) -> str:
        return _pelican_uri_to_local_cache_path(str(uri), cache_dir=self.cache_dir)

    def wait_for(self, uri: str) -> None:
        uri = str(uri)
        if uri not in self._status:
            return
        with self._cond:
            while True:
                st = self._status.get(uri)
                if st == "done":
                    return
                if st == "error":
                    raise RuntimeError(f"Prefetch failed for {uri}: {self._errors.get(uri, 'unknown error')}")
                if self._stop:
                    raise RuntimeError(f"Prefetch stopped before completing {uri}")
                self._cond.wait(timeout=0.5)

    def progress_string(self) -> str:
        """Human-friendly status for tqdm/postfix."""
        with self._cond:
            done = sum(1 for s in self._status.values() if s == "done")
            downloading = sum(1 for s in self._status.values() if s == "downloading")
            errors = sum(1 for s in self._status.values() if s == "error")
            total = len(self._status)
            # pending includes items not yet started.
            pending = total - done - downloading - errors
            cur = max(-1, self._current_index)
            nxt = self._next_download
        parts = [f"{done}/{total} cached"]
        if downloading:
            parts.append(f"{downloading} downloading")
        if pending:
            parts.append(f"{pending} pending")
        if errors:
            parts.append(f"{errors} errors")
        # Add a tiny bit of context about the rolling window.
        parts.append(f"cur={cur} next={nxt}")
        return ", ".join(parts)

    def _run(self) -> None:
        try:
            from pelicanfs.core import PelicanFileSystem
        except Exception as e:
            with self._cond:
                for u in self.pelican_uris:
                    self._status[u] = "error"
                    self._errors[u] = f"pelicanfs import failed: {e}"
                self._cond.notify_all()
            return

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        fs = PelicanFileSystem(self.federation_url, headers=headers)

        while True:
            with self._cond:
                if self._stop:
                    return

                target = min(len(self.pelican_uris), self._current_index + self.ahead + 1)
                if self._next_download >= target:
                    self._cond.wait(timeout=0.5)
                    continue

                uri = self.pelican_uris[self._next_download]
                self._next_download += 1

                # Mark as downloading.
                self._status[uri] = "downloading"
                self._cond.notify_all()

            out_path = self.local_path(uri)
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            try:
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    with self._cond:
                        self._status[uri] = "done"
                        self._cond.notify_all()
                    continue

                tmp_path = out_path + ".tmp"
                with fs.open(uri, "rb") as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                os.replace(tmp_path, out_path)
                with self._cond:
                    self._status[uri] = "done"
                    self._cond.notify_all()
            except Exception as e:
                with self._cond:
                    self._status[uri] = "error"
                    self._errors[uri] = repr(e)
                    self._cond.notify_all()


def _pelican_uri_dir_and_pattern(pelican_uri: str) -> tuple[str, str]:
    """Return (dir_uri, basename_pattern) for a pelican:// URI."""
    u = urlparse(pelican_uri)
    # Use posixpath for federation paths.
    base = posixpath.basename(u.path)
    dir_path = posixpath.dirname(u.path) or "/"
    dir_uri = f"pelican://{u.netloc}{dir_path}"
    return dir_uri, base


def expand_pelican_wildcards(
    infiles: list[str],
    *,
    federation_url: str | None,
    token: str | None,
) -> tuple[list[str], str | None]:
    """Expand pelican://... glob patterns into concrete file URIs.

    Returns (expanded_infiles, inferred_federation_url).
    """
    expanded: list[str] = []
    inferred_fed: str | None = federation_url

    # Lazily create FS only if we see a pelican wildcard.
    fs = None

    for item in infiles:
        item = str(item)
        if item.startswith("pelican://") and _has_wildcards(item):
            if inferred_fed is None:
                inferred_fed = _infer_pelican_federation_url(item)
            if inferred_fed is None:
                raise ValueError(
                    f"Could not infer --federation-url from pelican URI: {item}"
                )

            if fs is None:
                try:
                    from pelicanfs.core import PelicanFileSystem
                except ImportError as e:
                    raise ImportError(
                        "pelicanfs is required to expand pelican:// wildcards. Install with `pip install pelicanfs`."
                    ) from e
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                fs = PelicanFileSystem(inferred_fed, headers=headers)

            # Prefer glob() if supported by the filesystem.
            matches: list[str] = []
            glob_fn = getattr(fs, "glob", None)
            if callable(glob_fn):
                try:
                    res = glob_fn(item)
                    if isinstance(res, (list, tuple)):
                        matches = [str(x) for x in res]
                except Exception:
                    matches = []

            # Fallback: ls(dir) + fnmatch on basename.
            if not matches:
                dir_uri, basename_pat = _pelican_uri_dir_and_pattern(item)
                ls_fn = getattr(fs, "ls", None)
                if not callable(ls_fn):
                    raise RuntimeError(
                        "Pelican filesystem does not support glob() or ls(); cannot expand wildcards."
                    )
                entries = ls_fn(dir_uri)
                if entries and isinstance(entries[0], dict):
                    paths = [str(e.get("name", "")) for e in entries if e.get("name")]
                else:
                    paths = [str(e) for e in (entries or [])]
                matches = [p for p in paths if fnmatch.fnmatch(posixpath.basename(p), basename_pat)]

            # Normalize matches to full pelican:// URIs. Some filesystem backends
            # return plain paths like "/icecube/..." or "icecube/...".
            u = urlparse(item)
            normalized: list[str] = []
            for m in matches:
                m = str(m)
                if m.startswith("pelican://"):
                    normalized.append(m)
                    continue
                if not m.startswith("/"):
                    m = "/" + m
                normalized.append(f"pelican://{u.netloc}{m}")
            matches = normalized

            matches = sorted(set(matches))
            if not matches:
                raise FileNotFoundError(f"Pelican wildcard matched 0 files: {item}")
            expanded.extend(matches)
        else:
            expanded.append(item)

    return expanded, inferred_fed


def _is_pelican_path(p: str) -> bool:
    return str(p).startswith("pelican://")


def _infer_scope_path_from_pelican_uri(pelican_uri: str, *, storage_prefix: str = "/icecube/wipac") -> str:
    """Infer an authorization scope path from a pelican:// URI.

    We use the directory path as a prefix scope.
    """
    u = urlparse(pelican_uri)
    path = u.path or "/"

    # pelican:// URIs in this repo commonly look like:
    #   pelican://osg-htc.org/icecube/wipac/foo/bar
    # For token scopes we want permissions for /foo/bar.
    storage_prefix = (storage_prefix or "").rstrip("/")
    if storage_prefix and path.startswith(storage_prefix + "/"):
        path = path[len(storage_prefix) :]
        if not path:
            path = "/"

    # If the user explicitly provided a directory (trailing '/'), keep it as a directory
    # but normalize away the trailing slash in the returned scope.
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
        if not path:
            path = "/"

    # If the last path component looks like a filename or contains globs, scope to the directory.
    base = posixpath.basename(path)
    if _has_wildcards(base) or ("." in base and not base.endswith(".")):
        path = posixpath.dirname(path) or "/"

    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _first_batch_signature(
    prims_feats: torch.Tensor,
    real_muons_feats: torch.Tensor,
    counts: torch.Tensor,
    *,
    n_prims: int = 4,
    n_muons: int = 8,
    n_counts: int = 8,
) -> str:
    """Return a stable, lightweight signature for debugging progress.

    This is meant to answer: "am I actually seeing new data?" without dumping
    full batches. It hashes small slices of primaries/muons/counts.
    """

    def _as_bytes(x: torch.Tensor, max_rows: int) -> bytes:
        if not isinstance(x, torch.Tensor):
            return b""
        if x.numel() == 0:
            return b""
        # Keep it lightweight and deterministic.
        if x.dim() >= 2:
            x = x[: max_rows]
        else:
            x = x[: max_rows]
        x = x.detach().to("cpu")
        try:
            arr = x.contiguous().numpy()
        except Exception:
            # Fallback if numpy conversion fails for any reason.
            arr = x.contiguous().flatten().tolist()
            return repr(arr).encode("utf-8")
        return arr.tobytes()

    h = hashlib.sha1()
    h.update(_as_bytes(prims_feats, int(n_prims)))
    h.update(_as_bytes(real_muons_feats, int(n_muons)))
    h.update(_as_bytes(counts, int(n_counts)))

    # Include shapes so two different batches with same first values are less likely.
    h.update(repr(tuple(prims_feats.shape)).encode("utf-8"))
    h.update(repr(tuple(real_muons_feats.shape)).encode("utf-8"))
    h.update(repr(tuple(counts.shape)).encode("utf-8"))

    return h.hexdigest()[:12]


class _PrefetchIterator:
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


def _fetch_pelican_token_via_helper(
    *,
    scope_path: str,
    federation_url: str,
    oidc_url: str,
    auth_cache_file: str,
    storage_prefix: str,
    want_modify: bool = False,
) -> str:
    """Fetch an access token using the repo's device-flow logic (imported as a library).

    This avoids subprocess buffering issues and makes failures easier to debug.
    """
    import logging

    # Ensure device-flow instructions are visible.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        from utils.pelican.token_lib import get_access_token
    except ImportError as e:
        # Allow running when CWD is not repo root.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            from utils.pelican.token_lib import get_access_token
        except ImportError as e2:
            raise ImportError(
                "Auto-token requires rest-tools and the local token helper library. "
                "Make sure your environment includes the dependencies from requirements.txt."
            ) from e2

    # federation_url and storage_prefix are kept for backward compatibility with the CLI,
    # but token acquisition only needs scope paths and the issuer URL.
    _ = federation_url
    _ = storage_prefix

    return get_access_token(
        oidc_url=oidc_url,
        source_path=scope_path,
        target_path=scope_path,
        auth_cache_file=auth_cache_file,
        want_modify=want_modify,
    )


def _fs_put_json(fs, remote_path: str, data: dict) -> None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            json.dump(data, tmp)
        fs.put(tmp_path, remote_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _fs_put_torch_checkpoint(fs, remote_path: str, checkpoint_data: dict) -> None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(checkpoint_data, tmp_path)
        fs.put(tmp_path, remote_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _fs_put_file(fs, remote_path: str, local_path: str) -> None:
    """Upload a local file to a remote path via fs.put."""
    fs.put(str(local_path), str(remote_path))


def _select_torch_device(device_arg: str) -> torch.device:
    """Select torch device from CLI arg.

    device_arg:
      - "auto": prefer CUDA, then MPS, else CPU
      - "cuda": require CUDA
      - "mps": require Apple Metal (PyTorch MPS)
      - "cpu": force CPU
    """
    device_arg = (device_arg or "auto").lower()

    has_mps_backend = bool(getattr(torch.backends, "mps", None))
    mps_available = bool(torch.backends.mps.is_available()) if has_mps_backend else False
    rocm_build = bool(getattr(torch.version, "hip", None))

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    if device_arg == "rocm":
        # PyTorch uses the 'cuda' device type for ROCm as well.
        if not rocm_build:
            raise RuntimeError("--device rocm was requested, but this PyTorch build is not ROCm-enabled")
        if not torch.cuda.is_available():
            raise RuntimeError("--device rocm was requested, but no ROCm device is available")
        return torch.device("cuda")
    if device_arg == "mps":
        if not mps_available:
            raise RuntimeError("--device mps was requested, but MPS is not available")
        return torch.device("mps")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError(
        f"Unknown --device value: {device_arg}. Use one of: auto, cpu, cuda, rocm, mps"
    )


def _device_backend_label(device: torch.device) -> str:
    if device.type == "cuda":
        # CUDA device type may mean NVIDIA CUDA or AMD ROCm.
        if bool(getattr(torch.version, "hip", None)):
            return "rocm"
        return "cuda"
    return device.type


def _print_file_contents(file_path: str, fs=None, max_events: int = 5):
    """Print a compact preview of an input file.

    This is intended for debugging data/format issues during the file-iteration
    loop. It does not require the training model to be initialized.
    """
    max_events = int(max_events)
    if max_events <= 0:
        raise ValueError("max_events must be > 0")

    if file_path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except Exception as e:
            raise ImportError(
                "pyarrow is required to print parquet contents. Install with `pip install pyarrow`."
            ) from e

        if fs:
            with fs.open(file_path, 'rb') as f:
                pf = pq.ParquetFile(f)
                print(f"\n=== {file_path} (parquet) ===")
                print(f"row_groups={pf.num_row_groups} schema={pf.schema_arrow}")
                shown = 0
                for rg in range(pf.num_row_groups):
                    if shown >= max_events:
                        break
                    table = pf.read_row_group(rg)
                    pydict = table.to_pydict()
                    n = len(next(iter(pydict.values()))) if pydict else 0
                    for i in range(n):
                        if shown >= max_events:
                            break
                        primary = pydict.get('primary', [None])[i]
                        muons = pydict.get('muons', [None])[i]
                        maj = pydict.get('primary_major_id', [None])[i] if 'primary_major_id' in pydict else None
                        minr = pydict.get('primary_minor_id', [None])[i] if 'primary_minor_id' in pydict else None
                        mu_len = len(muons) if isinstance(muons, list) else (0 if muons is None else None)
                        first_mu = None
                        if isinstance(muons, list) and len(muons) > 0:
                            first_mu = muons[0]
                        if maj is not None and minr is not None:
                            print(f"event[{shown}] primary_ids=({maj},{minr}) primary={primary} n_muons={mu_len} first_muon={first_mu}")
                        else:
                            print(f"event[{shown}] primary={primary} n_muons={mu_len} first_muon={first_mu}")
                        shown += 1
        else:
            pf = pq.ParquetFile(file_path)
            print(f"\n=== {file_path} (parquet) ===")
            print(f"row_groups={pf.num_row_groups} schema={pf.schema_arrow}")
            shown = 0
            for rg in range(pf.num_row_groups):
                if shown >= max_events:
                    break
                table = pf.read_row_group(rg)
                pydict = table.to_pydict()
                n = len(next(iter(pydict.values()))) if pydict else 0
                for i in range(n):
                    if shown >= max_events:
                        break
                    primary = pydict.get('primary', [None])[i]
                    muons = pydict.get('muons', [None])[i]
                    maj = pydict.get('primary_major_id', [None])[i] if 'primary_major_id' in pydict else None
                    minr = pydict.get('primary_minor_id', [None])[i] if 'primary_minor_id' in pydict else None
                    mu_len = len(muons) if isinstance(muons, list) else (0 if muons is None else None)
                    first_mu = None
                    if isinstance(muons, list) and len(muons) > 0:
                        first_mu = muons[0]
                    if maj is not None and minr is not None:
                        print(f"event[{shown}] primary_ids=({maj},{minr}) primary={primary} n_muons={mu_len} first_muon={first_mu}")
                    else:
                        print(f"event[{shown}] primary={primary} n_muons={mu_len} first_muon={first_mu}")
                    shown += 1
        return

    # Default: HDF5
    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "h5py is required to print HDF5 contents. Install with `pip install h5py`."
        ) from e

    def _preview_h5(f):
        print(f"\n=== {file_path} (h5) ===")
        keys = list(f.keys())
        print(f"keys={keys}")
        prim = f.get('primaries', None)
        mu = f.get('muons', None)
        counts = f.get('counts', None)
        if prim is not None:
            print(f"primaries.shape={prim.shape} dtype={prim.dtype}")
        if mu is not None:
            print(f"muons.shape={mu.shape} dtype={mu.dtype}")
        if counts is not None:
            print(f"counts.shape={counts.shape} dtype={counts.dtype}")

        if prim is None or counts is None:
            return

        n_events = min(int(prim.shape[0]), int(counts.shape[0]), max_events)
        # Build offsets from counts so we can preview muon slices
        counts_arr = counts[:n_events]
        start = 0
        for i in range(n_events):
            c = int(counts_arr[i])
            p = prim[i]
            m_slice = mu[start:start + c] if (mu is not None and c > 0) else None
            first_mu = None
            if m_slice is not None and len(m_slice) > 0:
                first_mu = m_slice[0].tolist() if hasattr(m_slice[0], 'tolist') else m_slice[0]
            print(f"event[{i}] primary={p.tolist() if hasattr(p,'tolist') else p} count={c} first_muon={first_mu}")
            start += c

    if fs:
        with fs.open(file_path, 'rb') as remote_f:
            with h5py.File(remote_f, 'r') as f:
                _preview_h5(f)
    else:
        with h5py.File(file_path, 'r') as f:
            _preview_h5(f)

def get_filesystem(federation_url, token):
    if federation_url:
        try:
            from pelicanfs.core import PelicanFileSystem
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            return PelicanFileSystem(federation_url, headers=headers)
        except ImportError:
            print("Warning: pelicanfs not found, falling back to local filesystem")
            return None
    return None


def _select_checkpoint_fs(path: str | None, *, fs, mode: str) -> object | None:
    """Select filesystem for checkpoint/model-checkpoint IO.

    mode:
      - "auto": use fs only for pelican:// paths
      - "local": always use local filesystem
      - "pelican": always use provided fs (if any)
    """
    if not path:
        return None
    mode = (mode or "auto").lower()
    if mode == "local":
        return None
    if mode == "pelican":
        return fs
    # auto
    return fs if _is_pelican_path(str(path)) else None

def load_progress(checkpoint_path, fs=None) -> tuple[int, set[str]]:
    """Load progress tracking.

    Backward compatible:
      - Old format: {"processed_files": [...]} (implicit epoch=0)
      - New format: {"epoch": int, "processed_files": [...]} (epoch is the current epoch)
    """
    processed_files: set[str] = set()
    epoch = 0

    def _parse(data: object) -> None:
        nonlocal epoch, processed_files
        if isinstance(data, dict):
            epoch = int(data.get("epoch", 0) or 0)
            processed_files = set(data.get("processed_files", []) or [])

    if fs:
        try:
            if fs.exists(checkpoint_path):
                with fs.open(checkpoint_path, 'r') as f:
                    _parse(json.load(f))
        except Exception as e:
            print(f"Warning: Could not read checkpoint from Pelican: {e}")
    elif checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            try:
                _parse(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode checkpoint file {checkpoint_path}")

    return epoch, processed_files

def save_progress(checkpoint_path, epoch: int, processed_files: set[str], fs=None) -> None:
    if not checkpoint_path:
        return
    payload = {"epoch": int(epoch), "processed_files": list(processed_files)}
    if fs:
        try:
            _fs_put_json(fs, checkpoint_path, payload)
        except Exception as e:
            print(f"Warning: Could not save checkpoint to Pelican: {e}")
    else:
        with open(checkpoint_path, 'w') as f:
            json.dump(payload, f)

def save_model_checkpoint(path, gen, crit, opt_G, opt_C, epoch=0, fs=None):
    checkpoint_data = {
        'gen_state_dict': gen.state_dict(),
        'crit_state_dict': crit.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_C_state_dict': opt_C.state_dict(),
        'epoch': epoch
    }
    
    if fs:
        try:
            _fs_put_torch_checkpoint(fs, path, checkpoint_data)
            print(f"Model checkpoint saved to Pelican: {path}")
        except Exception as e:
            print(f"Warning: Could not save model checkpoint to Pelican: {e}")
    else:
        torch.save(checkpoint_data, path)
        print(f"Model checkpoint saved to {path}")

def load_model_checkpoint(path, gen, crit, opt_G, opt_C, device, fs=None):
    checkpoint = None
    if fs:
        try:
            if fs.exists(path):
                with fs.open(path, 'rb') as f:
                    checkpoint = torch.load(f, map_location=device)
                    print(f"Model checkpoint loaded from Pelican: {path}")
        except Exception as e:
            print(f"Warning: Could not load model checkpoint from Pelican: {e}")
    elif path and os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        print(f"Model checkpoint loaded from {path}")

    if checkpoint:
        gen.load_state_dict(checkpoint['gen_state_dict'])
        crit.load_state_dict(checkpoint['crit_state_dict'])
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_C.load_state_dict(checkpoint['opt_C_state_dict'])
        return checkpoint.get('epoch', 0)
    return 0

def main(args):
    # Normalize inputs
    raw_infiles = [str(x) for x in args.infiles]

    checkpoint_paths: list[str] = []
    if getattr(args, "checkpoint", None):
        checkpoint_paths.append(str(args.checkpoint))
    if getattr(args, "model_checkpoint", None):
        checkpoint_paths.append(str(args.model_checkpoint))

    # Infer federation URL from any pelican:// path if not provided.
    inferred_fed = args.federation_url
    if inferred_fed is None:
        for p in (raw_infiles + checkpoint_paths):
            if _is_pelican_path(p):
                inferred_fed = _infer_pelican_federation_url(p)
                break
    if args.federation_url is None and inferred_fed is not None:
        args.federation_url = inferred_fed

    # If pelican paths are present but no token flow is enabled, warn early.
    # Prefer checkpoint paths for scope inference (they may require write access).
    pelican_paths_all = [p for p in (checkpoint_paths + raw_infiles) if _is_pelican_path(p)]
    pelican_checkpoint_paths = [p for p in checkpoint_paths if _is_pelican_path(p)]
    if pelican_paths_all and (args.token is None) and (not args.auto_token):
        print(
            "Warning: pelican:// paths detected but no --token provided and --auto-token is not set. "
            "If the data is not public, rerun with --auto-token (device flow) or provide --token."
        )

    # Optional: fetch token if pelican inputs are present but token is missing.
    if args.auto_token and (args.token is None):
        pelican_paths = pelican_paths_all
        if pelican_paths:
            if args.federation_url is None:
                raise ValueError(
                    "--auto-token was set but --federation-url could not be inferred. "
                    "Provide --federation-url explicitly."
                )

            scope_path = args.pelican_scope_path
            if not scope_path:
                scope_path = _infer_scope_path_from_pelican_uri(
                    pelican_paths[0],
                    storage_prefix=args.pelican_storage_prefix,
                )

            print(f"Fetching Pelican token for scope: {scope_path}")
            args.token = _fetch_pelican_token_via_helper(
                scope_path=scope_path,
                federation_url=args.federation_url,
                oidc_url=args.pelican_oidc_url,
                auth_cache_file=args.pelican_auth_cache_file,
                storage_prefix=args.pelican_storage_prefix,
            )

    # Expand pelican:// wildcards after token is available (if needed)
    expanded_infiles, inferred_fed2 = expand_pelican_wildcards(
        raw_infiles,
        federation_url=args.federation_url,
        token=args.token,
    )
    args.infiles = expanded_infiles
    if args.federation_url is None and inferred_fed2 is not None:
        args.federation_url = inferred_fed2

    # Initialize Filesystem for (optional) remote reads
    fs = get_filesystem(args.federation_url, args.token)

    # Checkpoint/model-checkpoint IO may be local even when inputs are pelican://
    checkpoint_fs = _select_checkpoint_fs(args.checkpoint, fs=fs, mode=getattr(args, "checkpoint_io", "auto"))
    model_checkpoint_fs = _select_checkpoint_fs(
        args.model_checkpoint,
        fs=fs,
        mode=getattr(args, "checkpoint_io", "auto"),
    )

    # Print-only mode: preview file contents and exit
    if args.print_file_contents:
        for p in [str(x) for x in args.infiles]:
            _print_file_contents(p, fs=fs, max_events=args.print_max_events)
        return

    device = _select_torch_device(args.device)
    print(f"Using device: {device} ({_device_backend_label(device)})")

    writer = None
    tb_dir = None
    run_name = None
    if getattr(args, "tb_logdir", None):
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requested (--tb-logdir) but torch.utils.tensorboard is unavailable. "
                "Install with: pip install tensorboard"
            )
        run_name = getattr(args, "tb_run_name", None) or time.strftime("%Y%m%d-%H%M%S")
        tb_dir = os.path.join(str(args.tb_logdir), str(run_name))
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        try:
            writer.add_text("run/args", json.dumps(vars(args), indent=2, default=str), 0)
        except Exception:
            pass
        print(f"TensorBoard logging enabled: {tb_dir}")

    # Optional: sync TB event files to another location (local dir or pelican://)
    tb_sync_to = getattr(args, "tb_sync_to", None)
    tb_sync_fs = None
    tb_sync_base = None
    tb_uploaded: dict[str, tuple[float, int]] = {}
    tb_last_sync_t = 0.0

    if tb_sync_to is not None:
        if writer is None or tb_dir is None or run_name is None:
            raise ValueError("--tb-sync-to requires --tb-logdir to be set")

        tb_sync_fs = _select_checkpoint_fs(tb_sync_to, fs=fs, mode=getattr(args, "tb_io", "auto"))
        if tb_sync_fs is None:
            # Local destination
            tb_sync_base = os.path.join(str(tb_sync_to), str(run_name))
            os.makedirs(tb_sync_base, exist_ok=True)
        else:
            # Remote destination (pelican://). Treat tb_sync_to as a prefix and mirror under run_name.
            base = str(tb_sync_to).rstrip("/")
            tb_sync_base = posixpath.join(base, str(run_name))

        print(f"TensorBoard sync enabled: {tb_sync_to} (interval={getattr(args, 'tb_sync_interval', 60.0)}s)")

    def _tb_sync(force: bool = False) -> None:
        nonlocal tb_last_sync_t
        if writer is None or tb_dir is None or tb_sync_to is None or tb_sync_base is None:
            return

        interval_s = float(getattr(args, "tb_sync_interval", 60.0) or 0.0)
        now = time.perf_counter()
        if (not force) and interval_s > 0 and (now - tb_last_sync_t) < interval_s:
            return

        tb_last_sync_t = now
        try:
            writer.flush()
        except Exception:
            pass

        for root, _, files in os.walk(tb_dir):
            for name in files:
                local_path = os.path.join(root, name)
                try:
                    st = os.stat(local_path)
                except OSError:
                    continue

                rel = os.path.relpath(local_path, tb_dir)
                prev = tb_uploaded.get(rel)
                sig = (float(st.st_mtime), int(st.st_size))
                if prev is not None and prev == sig:
                    continue

                if tb_sync_fs is None:
                    dest_path = os.path.join(tb_sync_base, rel)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    try:
                        shutil.copy2(local_path, dest_path)
                        tb_uploaded[rel] = sig
                    except Exception:
                        continue
                else:
                    dest_path = posixpath.join(str(tb_sync_base), rel.replace(os.sep, "/"))
                    try:
                        _fs_put_file(tb_sync_fs, dest_path, local_path)
                        tb_uploaded[rel] = sig
                    except Exception:
                        continue

    # Initialize Models
    gen = ScalableGenerator(
        cond_dim=args.cond_dim, 
        feat_dim=args.feat_dim, 
        latent_dim_global=args.latent_dim_global, 
        latent_dim_local=args.latent_dim_local, 
        hidden_dim=args.hidden_dim,
        device=device
    ).to(device)
    
    crit = ScalableCritic(
        feat_dim=args.feat_dim, 
        cond_dim=args.cond_dim, 
        device=device
    ).to(device)

    # Optional: torch.compile (PyTorch 2.x). This can improve GPU utilization by
    # reducing Python overhead / fusing kernels, at the cost of compile warm-up.
    # Note: WGAN-GP uses a gradient penalty that requires higher-order gradients
    # ("double backward"), which is not currently supported by torch.compile's
    # default AOTAutograd path. By default we therefore do NOT compile the critic
    # when --lambda-gp>0.
    base_mode = str(getattr(args, "torch_compile", "off") or "off").lower()
    gen_mode = str(getattr(args, "torch_compile_gen", "auto") or "auto").lower()
    crit_mode = str(getattr(args, "torch_compile_critic", "auto") or "auto").lower()

    if gen_mode == "auto":
        gen_mode = base_mode
    if crit_mode == "auto":
        crit_mode = base_mode

    allowed_modes = {"off", "default", "reduce-overhead", "max-autotune"}
    if base_mode not in allowed_modes:
        raise ValueError(f"Invalid --torch-compile mode: {base_mode}")
    if gen_mode not in allowed_modes:
        raise ValueError(f"Invalid --torch-compile-gen mode: {gen_mode}")
    if crit_mode not in allowed_modes:
        raise ValueError(f"Invalid --torch-compile-critic mode: {crit_mode}")

    lambda_gp = float(getattr(args, "lambda_gp", 0.0) or 0.0)
    if lambda_gp > 0.0 and crit_mode != "off":
        # If user explicitly requested critic compilation, fail fast with a clear error.
        explicitly_requested = str(getattr(args, "torch_compile_critic", "auto") or "auto").lower() != "auto"
        if explicitly_requested:
            raise RuntimeError(
                "--torch-compile-critic is not supported when --lambda-gp>0 (WGAN-GP requires double backward, "
                "which torch.compile/aot_autograd does not support). "
                "Set --torch-compile-critic off, or set --lambda-gp 0 if you really want to compile the critic."
            )
        # Auto mode: silently force off but explain once.
        print(
            f"Info: forcing critic compilation off because --lambda-gp={args.lambda_gp} requires double backward (unsupported by torch.compile)."
        )
        crit_mode = "off"

    if (gen_mode != "off" or crit_mode != "off"):
        if not hasattr(torch, "compile"):
            print("Warning: torch.compile requested but this PyTorch does not support torch.compile; ignoring.")
        else:
            try:
                def _compile(m, mode: str):
                    mode_arg = None if mode == "default" else mode
                    return torch.compile(m) if mode_arg is None else torch.compile(m, mode=mode_arg)

                msg_parts = []
                if gen_mode != "off":
                    msg_parts.append(f"gen={gen_mode}")
                if crit_mode != "off":
                    msg_parts.append(f"crit={crit_mode}")
                print("torch.compile enabled: " + ", ".join(msg_parts))

                if gen_mode != "off":
                    gen = _compile(gen, gen_mode)
                if crit_mode != "off":
                    crit = _compile(crit, crit_mode)
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}); continuing without compilation.")
    
    opt_G = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_C = optim.Adam(crit.parameters(), lr=1e-4, betas=(0.0, 0.9))

    # Initialize Filesystem for Checkpoints
    # (already initialized above)

    # Load Checkpoints
    progress_epoch, processed_files = load_progress(args.checkpoint, fs=checkpoint_fs)
    start_epoch = load_model_checkpoint(
        args.model_checkpoint,
        gen,
        crit,
        opt_G,
        opt_C,
        device,
        fs=model_checkpoint_fs,
    )
    
    # Convert input paths to strings for consistent comparison
    all_files = [str(p) for p in args.infiles]
    files_to_process = [f for f in all_files if f not in processed_files]
    
    print(f"Total files: {len(all_files)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Remaining: {len(files_to_process)}")

    # Optional: prefetch pelican:// inputs to local disk.
    prefetcher: _PelicanPrefetcher | None = None
    if getattr(args, "prefetch_dir", None):
        pelican_candidates = [str(p) for p in files_to_process if _is_pelican_path(p)]
        max_files = int(getattr(args, "prefetch_max_files", 0) or 0)
        if max_files > 0:
            pelican_candidates = pelican_candidates[:max_files]

        if pelican_candidates:
            if fs is None:
                raise RuntimeError(
                    "--prefetch-dir was set but PelicanFS is not configured. "
                    "Provide --federation-url (or pelican:// inputs) and ensure pelicanfs is installed."
                )

            ahead = int(getattr(args, "prefetch_ahead", 0) or 0)
            if ahead > 0:
                print(
                    f"Starting background prefetch: ahead={ahead}, max_files={max_files or 'all'}, cache={args.prefetch_dir}"
                )
                prefetcher = _PelicanPrefetcher(
                    pelican_candidates,
                    federation_url=args.federation_url,
                    token=args.token,
                    cache_dir=args.prefetch_dir,
                    ahead=ahead,
                )
                prefetcher.start()
            else:
                print(
                    f"Prefetching {len(pelican_candidates)} pelican:// files to {args.prefetch_dir} (blocking)"
                )
                _prefetch_pelican_files(
                    pelican_candidates,
                    fs=fs,
                    cache_dir=args.prefetch_dir,
                )
                print(f"Prefetch complete. Training will read from: {args.prefetch_dir}")

    normalizer = DataNormalizer()

    memory_cache = None
    mem_cache_mb = int(getattr(args, "memory_cache_mb", 0) or 0)
    if mem_cache_mb > 0:
        memory_cache = FileBytesLRUCache(max_bytes=mem_cache_mb * 1024 * 1024)
        print(f"In-memory parquet cache enabled: {mem_cache_mb} MiB (process-local)")

    train_steps_done = 0
    file_pbar = tqdm(files_to_process, desc="Files", unit="file")
    for file_idx, file_path in enumerate(file_pbar, start=1):
        file_pbar.set_description(f"Processing {os.path.basename(file_path)}")

        if prefetcher is not None:
            file_pbar.set_postfix(prefetch=prefetcher.progress_string())

        # In non-interactive environments, tqdm may not render well. Emit a
        # clear per-file marker to stdout.
        tqdm.write(f"[file {file_idx}/{len(files_to_process)}] start source={file_path}")

        read_path = file_path
        cached_local_path = None
        if getattr(args, "prefetch_dir", None) and _is_pelican_path(file_path):
            cached = _pelican_uri_to_local_cache_path(file_path, cache_dir=args.prefetch_dir)
            cached_local_path = cached
            if prefetcher is not None and file_path in prefetcher.uri_to_index:
                prefetcher.update_current_uri(file_path)
                file_pbar.set_postfix(prefetch=prefetcher.progress_string())
                prefetcher.wait_for(file_path)
                file_pbar.set_postfix(prefetch=prefetcher.progress_string())
                read_path = cached
            elif os.path.exists(cached) and os.path.getsize(cached) > 0:
                read_path = cached

        if read_path != file_path:
            tqdm.write(f"[file {file_idx}/{len(files_to_process)}] read_path={read_path} (cached)")
        else:
            tqdm.write(f"[file {file_idx}/{len(files_to_process)}] read_path={read_path}")

        # Fail fast with a more actionable message than a deep h5py traceback.
        if not _is_pelican_path(read_path) and not os.path.exists(read_path):
            raise FileNotFoundError(
                f"Input file not found: {read_path}\n"
                "If this is a host filesystem path (e.g. /icecube/...), make sure it is available/mounted in your environment.\n"
                "If you intended to read via Pelican, pass a pelican:// URI (and optionally --federation-url/--token)."
            )

        file_format = _infer_file_format(read_path)

        fed_for_read = args.federation_url if _is_pelican_path(read_path) else None
        token_for_read = args.token if _is_pelican_path(read_path) else None

        # Parquet and pelican:// inputs require the HF streaming loader in this repo.
        use_hf_for_file = bool(args.use_hf) or file_format == "parquet" or _is_pelican_path(read_path)
        if (not args.use_hf) and use_hf_for_file and (file_format == "parquet" or _is_pelican_path(read_path)):
            print(
                f"Info: using HF streaming loader for {file_format} input: {read_path} "
                "(add --use-hf to enable this explicitly)."
            )

        tqdm.write(
            f"[file {file_idx}/{len(files_to_process)}] loader={'hf_streaming' if use_hf_for_file else 'hdf5_local'} format={file_format}"
        )
        
        if use_hf_for_file:
            # Fast-path for parquet: yield already-batched tensors directly from Arrow
            # record batches to reduce Python/HF collation overhead.
            if file_format == "parquet" and bool(getattr(args, "parquet_batch_reader", False)):
                dataset = get_parquet_batch_dataset(
                    [read_path],
                    batch_size=args.batch_size,
                    federation_url=fed_for_read,
                    token=token_for_read,
                    memory_cache=memory_cache,
                )
                collate = None
            else:
                dataset = get_hf_dataset(
                    [read_path],
                    file_format=file_format,
                    streaming=True,
                    federation_url=fed_for_read,
                    token=token_for_read,
                    memory_cache=memory_cache,
                )
                collate = hf_collate_fn
        else:
            # Local HDF5 fast-path.
            dataset = SingleHDF5Dataset(read_path)
            collate = ragged_collate_fn

        num_workers = int(getattr(args, "num_workers", 0) or 0)
        if (not use_hf_for_file) and num_workers > 0:
            print(
                "Warning: --num-workers>0 is not supported for local HDF5 fast-path; forcing --num-workers=0 for this file."
            )
            num_workers = 0

        if collate is not None and num_workers > 0:
            print(
                "Warning: --num-workers>0 is currently only supported with --parquet-batch-reader (already-batched dataset). "
                "Forcing --num-workers=0 for this file."
            )
            num_workers = 0

        dl_kwargs = {
            "num_workers": num_workers,
            "pin_memory": bool(getattr(args, "pin_memory", False)),
        }
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2) or 2)
            dl_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", False))

        if collate is None:
            # Dataset already yields fully-formed batches.
            dataloader = DataLoader(dataset, batch_size=None, **dl_kwargs)
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=collate,
                **dl_kwargs,
            )

        # For streaming datasets, measuring how long we spend waiting for the
        # next batch (I/O + decode + collate) vs training compute is useful.
        data_iter = iter(dataloader)
        batch_prefetch = int(getattr(args, "prefetch_batches", 0) or 0)
        if batch_prefetch > 0:
            data_iter = _PrefetchIterator(data_iter, max_prefetch=batch_prefetch)
        batch_pbar = tqdm(desc="Batches", unit="batch", leave=False)
        batches_seen = 0
        events_seen = 0
        muons_seen = 0
        skipped_empty = 0
        reported_first_batch = False

        file_t0 = time.perf_counter()
        load_time_s = 0.0
        step_time_s = 0.0

        while True:
            t_load0 = time.perf_counter()
            try:
                real_muons, batch_idx, prims, counts = next(data_iter)
            except StopIteration:
                break
            t_load1 = time.perf_counter()
            load_time_s += (t_load1 - t_load0)

            batches_seen += 1
            batch_pbar.update(1)

            events_seen += int(counts.numel())
            muons_seen += int(counts.sum().item())
            # Handle IDs if present (from HF dataloader/Parquet)
            # Primaries: [Batch, 6] -> [Batch, 4] (Skip first 2)
            if prims.shape[1] == 6:
                prims_feats = prims[:, 2:]
            else:
                prims_feats = prims
                
            # Muons: [Total, 5] -> [Total, 3] (Skip first 2)
            if real_muons.shape[1] == 5:
                real_muons_feats = real_muons[:, 2:]
            else:
                real_muons_feats = real_muons

            

            t_step0 = time.perf_counter()

            # Optional: report a small signature so you can confirm data changes across files.
            if (not reported_first_batch) and bool(getattr(args, "report_first_batch", False)):
                if real_muons_feats.numel() > 0 and int(counts.sum().item()) > 0:
                    sig = _first_batch_signature(prims_feats, real_muons_feats, counts)
                    c_preview = counts[:8].detach().to("cpu").tolist()
                    tqdm.write(
                        f"[file {file_idx}/{len(files_to_process)}] first_batch signature={sig} "
                        f"events={int(counts.numel())} muons={int(counts.sum().item())} counts[:8]={c_preview}"
                    )
                    reported_first_batch = True

            # Move to device. With pinned memory (see --pin-memory) these can be non-blocking.
            non_blocking = bool(getattr(args, "pin_memory", False)) and str(device).startswith("cuda")
            real_muons_feats = real_muons_feats.to(device, non_blocking=non_blocking)
            batch_idx = batch_idx.to(device, non_blocking=non_blocking)
            prims_feats = prims_feats.to(device, non_blocking=non_blocking)
            counts = counts.to(device, non_blocking=non_blocking)

            # Some events (or entire batches) can have zero muons. WGAN training
            # requires at least one real sample to compute losses and gradient penalty.
            if real_muons_feats.numel() == 0 or int(counts.sum().item()) == 0:
                skipped_empty += 1
                continue

            # Normalize
            real_muons_norm = normalizer.normalize_features(real_muons_feats)
            prims_norm = normalizer.normalize_primaries(prims_feats)
            
            c_loss, g_loss = train_step_scalable(
                gen, crit, opt_G, opt_C,
                real_muons_norm, batch_idx, prims_norm, counts,
                lambda_gp=args.lambda_gp,
                critic_steps=int(getattr(args, "critic_steps", 1) or 1),
                device=device
            )

            t_step1 = time.perf_counter()
            step_time_s += (t_step1 - t_step0)

            train_steps_done += 1
            
            batch_pbar.set_postfix(c_loss=f"{c_loss:.4f}", g_loss=f"{g_loss:.4f}")

            # TensorBoard logging (optional)
            if writer is not None:
                tb_every = int(getattr(args, "tb_log_interval", 0) or 0)
                if tb_every > 0 and (train_steps_done % tb_every == 0):
                    elapsed = max(1e-9, time.perf_counter() - file_t0)
                    avg_load_ms = (load_time_s / max(1, batches_seen)) * 1e3
                    avg_step_ms = (step_time_s / max(1, batches_seen)) * 1e3
                    writer.add_scalar("train/c_loss", float(c_loss), train_steps_done)
                    writer.add_scalar("train/g_loss", float(g_loss), train_steps_done)
                    writer.add_scalar("perf/avg_load_ms", float(avg_load_ms), train_steps_done)
                    writer.add_scalar("perf/avg_step_ms", float(avg_step_ms), train_steps_done)
                    writer.add_scalar("perf/batch_per_s", float(batches_seen / elapsed), train_steps_done)
                    writer.add_scalar("perf/events_per_s", float(events_seen / elapsed), train_steps_done)
                    writer.add_scalar("perf/muons_per_s", float(muons_seen / elapsed), train_steps_done)
                    writer.add_scalar("data/events_seen", float(events_seen), train_steps_done)
                    writer.add_scalar("data/muons_seen", float(muons_seen), train_steps_done)
                    writer.add_scalar(
                        "data/mean_muons_per_event",
                        float(muons_seen / max(1, events_seen)),
                        train_steps_done,
                    )
                    try:
                        writer.add_scalar("data/mean_counts", float(counts.float().mean().item()), train_steps_done)
                        writer.add_scalar("data/max_counts", float(counts.max().item()), train_steps_done)
                    except Exception:
                        pass

                    # Periodically sync event files if requested.
                    try:
                        _tb_sync(force=False)
                    except Exception:
                        pass

                hist_every = int(getattr(args, "tb_hist_interval", 0) or 0)
                if hist_every > 0 and (train_steps_done % hist_every == 0):
                    max_mu = int(getattr(args, "tb_max_muons", 200000) or 200000)

                    # Counts + primaries
                    try:
                        writer.add_histogram("data/counts", counts.detach().to("cpu"), train_steps_done)
                    except Exception:
                        pass
                    try:
                        p_cpu = prims_norm.detach().to("cpu")
                        for d in range(min(int(p_cpu.shape[1]), 16)):
                            writer.add_histogram(f"data/primaries_dim{d}", p_cpu[:, d], train_steps_done)
                    except Exception:
                        pass

                    # Real vs fake muon feature histograms (normalized space)
                    try:
                        real_cpu = real_muons_norm.detach()
                        if real_cpu.is_cuda:
                            real_cpu = real_cpu[:max_mu].to("cpu")
                        else:
                            real_cpu = real_cpu[:max_mu]
                        for d in range(min(int(real_cpu.shape[1]), 8)):
                            writer.add_histogram(f"real/muon_feat{d}", real_cpu[:, d], train_steps_done)
                    except Exception:
                        pass

                    try:
                        with torch.no_grad():
                            fake_mu, _ = gen(prims_norm, counts)
                        fake_cpu = fake_mu.detach()
                        if fake_cpu.is_cuda:
                            fake_cpu = fake_cpu[:max_mu].to("cpu")
                        else:
                            fake_cpu = fake_cpu[:max_mu]
                        if fake_cpu.numel() > 0:
                            for d in range(min(int(fake_cpu.shape[1]), 8)):
                                writer.add_histogram(f"fake/muon_feat{d}", fake_cpu[:, d], train_steps_done)
                    except Exception:
                        pass

            # Always print something occasionally for streaming datasets.
            log_every = int(getattr(args, "log_interval", 0) or 0)
            if log_every > 0 and (batches_seen % log_every == 0):
                elapsed = max(1e-9, time.perf_counter() - file_t0)
                bps = batches_seen / elapsed
                eps = events_seen / elapsed
                mps = muons_seen / elapsed
                avg_load_ms = (load_time_s / max(1, batches_seen)) * 1e3
                avg_step_ms = (step_time_s / max(1, batches_seen)) * 1e3
                tqdm.write(
                    f"[file {file_idx}/{len(files_to_process)}] batches={batches_seen} "
                    f"events={events_seen} muons={muons_seen} "
                    f"skipped_empty={skipped_empty} "
                    f"rate: {bps:.2f} batch/s {eps:.1f} evt/s {mps:.1f} mu/s "
                    f"avg: load={avg_load_ms:.1f}ms step={avg_step_ms:.1f}ms "
                    f"c_loss={c_loss:.4f} g_loss={g_loss:.4f}"
                )

        batch_pbar.close()

        # Checkpoint after file is done
        processed_files.add(file_path)
        # This script currently runs a single pass over files. We store epoch=0
        # for forward compatibility if you later add an epoch loop.
        save_progress(args.checkpoint, progress_epoch, processed_files, fs=checkpoint_fs)
        save_model_checkpoint(args.model_checkpoint, gen, crit, opt_G, opt_C, fs=model_checkpoint_fs)

        tqdm.write(
            f"[file {file_idx}/{len(files_to_process)}] done batches={batches_seen} events={events_seen} muons={muons_seen} skipped_empty={skipped_empty}"
        )

        # Optionally delete cached prefetch-dir copy after this file is fully consumed.
        if bool(getattr(args, "prefetch_delete_after_use", False)) and cached_local_path and (read_path == cached_local_path):
            try:
                if os.path.exists(cached_local_path):
                    os.remove(cached_local_path)
                    tqdm.write(f"[file {file_idx}/{len(files_to_process)}] deleted cache={cached_local_path}")
            except Exception as e:
                tqdm.write(f"[file {file_idx}/{len(files_to_process)}] warning: could not delete cache={cached_local_path}: {e}")

    if prefetcher is not None:
        prefetcher.stop()

    if writer is not None:
        try:
            writer.flush()
        except Exception:
            pass
        try:
            _tb_sync(force=True)
        except Exception:
            pass
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--infiles",
        nargs="+",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Use Hugging Face Streaming Dataset",
    )
    parser.add_argument(
        "--parquet-batch-reader",
        action="store_true",
        help=(
            "For parquet inputs, read Arrow record batches directly and yield already-batched tensors "
            "(bypasses HuggingFace per-example conversion/collate; usually much faster)."
        ),
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=0,
        help=(
            "Prefetch up to N already-batched training batches in a background thread to overlap input decoding with GPU compute. "
            "0 disables (default: 0)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "PyTorch DataLoader worker processes for parallel input decode. "
            "For parquet streaming/--parquet-batch-reader this can reduce avg load time (default: 0)."
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker when --num-workers>0 (default: 2).",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive across batches (only when --num-workers>0).",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Use pinned host memory for faster host->GPU transfers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for the PyTorch DataLoader (default: 1024).",
    )
    parser.add_argument(
        "--critic-steps",
        type=int,
        default=1,
        help="Number of critic updates per generator update (WGAN-GP). Default: 1.",
    )
    parser.add_argument(
        "--torch-compile",
        type=str,
        default="off",
        choices=["off", "default", "reduce-overhead", "max-autotune"],
        help="Enable torch.compile for generator+critic (PyTorch 2.x). Default: off.",
    )
    parser.add_argument(
        "--torch-compile-gen",
        type=str,
        default="auto",
        choices=["auto", "off", "default", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode for generator. 'auto' uses --torch-compile. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--torch-compile-critic",
        type=str,
        default="auto",
        choices=["auto", "off", "default", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode for critic. 'auto' uses --torch-compile, but will be forced off when --lambda-gp>0 "
            "(double backward not supported). Default: auto."
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print a progress line every N batches (default: 50, set 0 to disable).",
    )
    parser.add_argument(
        "--tb-logdir",
        type=str,
        default=None,
        help="If set, write TensorBoard logs under this directory (disabled by default).",
    )
    parser.add_argument(
        "--tb-run-name",
        type=str,
        default=None,
        help="Optional run subdirectory name under --tb-logdir (default: auto timestamp).",
    )
    parser.add_argument(
        "--tb-log-interval",
        type=int,
        default=50,
        help="Write TensorBoard scalars every N batches (default: 50).",
    )
    parser.add_argument(
        "--tb-hist-interval",
        type=int,
        default=0,
        help="Write TensorBoard histograms every N batches (0 disables; default: 0).",
    )
    parser.add_argument(
        "--tb-max-muons",
        type=int,
        default=200000,
        help="Max muons to include in a histogram dump (default: 200000).",
    )
    parser.add_argument(
        "--tb-sync-to",
        type=str,
        default=None,
        help=(
            "Optional destination to periodically sync TensorBoard event files to. "
            "Can be a local directory or a pelican:// URI prefix."
        ),
    )
    parser.add_argument(
        "--tb-sync-interval",
        type=float,
        default=60.0,
        help="Seconds between TensorBoard syncs when --tb-sync-to is set (default: 60).",
    )
    parser.add_argument(
        "--tb-io",
        type=str,
        default="auto",
        choices=["auto", "local", "pelican"],
        help="Where to write TensorBoard sync output (auto/local/pelican). Default: auto.",
    )
    parser.add_argument(
        "--report-first-batch",
        action="store_true",
        help=(
            "Print a one-time per-file summary of the first non-empty batch (counts preview + a small hash signature). "
            "Useful to confirm new data is being consumed across files when streaming."
        ),
    )
    parser.add_argument(
        "--memory-cache-mb",
        type=int,
        default=0,
        help=(
            "Cache Parquet file bytes in RAM up to this many MiB (LRU, process-local). "
            "Helps when Parquet decode/open is the bottleneck. 0 disables (default: 0)."
        ),
    )
    parser.add_argument(
        "--prefetch-dir",
        type=str,
        default=None,
        help=(
            "If set, download pelican:// input files into this local directory before training and read from the cached copies. "
            "Useful for reducing repeated remote reads."
        ),
    )
    parser.add_argument(
        "--prefetch-delete-after-use",
        action="store_true",
        help="If set, delete each cached file under --prefetch-dir after it has been fully used for training.",
    )
    parser.add_argument(
        "--prefetch-max-files",
        type=int,
        default=0,
        help=(
            "Max number of pelican:// input files to prefetch when --prefetch-dir is set. "
            "0 means prefetch all (default: 0)."
        ),
    )
    parser.add_argument(
        "--prefetch-ahead",
        type=int,
        default=0,
        help=(
            "If >0 and --prefetch-dir is set, prefetch pelican:// files in a background thread while training runs, "
            "keeping up to N files ahead cached. 0 disables background prefetch (default: 0)."
        ),
    )
    parser.add_argument(
        "--federation-url",
        type=str,
        default=None,
        help="Pelican Federation URL (e.g. pelican://osg-htc.org)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Auth token for Pelican",
    )
    parser.add_argument(
        "--auto-token",
        action="store_true",
        help=(
            "If pelican:// inputs are provided and --token is omitted, fetch a token via "
            "utils/pelican/get_pelican_token.py (device flow)."
        ),
    )
    parser.add_argument(
        "--pelican-scope-path",
        type=str,
        default=None,
        help=(
            "Scope path to request token permissions for (passed as both --source-path and --target-path). "
            "If omitted, inferred from the first pelican:// infile."
        ),
    )
    parser.add_argument(
        "--pelican-oidc-url",
        type=str,
        default="https://token-issuer.icecube.aq",
        help="OIDC issuer URL for device flow (default: https://token-issuer.icecube.aq).",
    )
    parser.add_argument(
        "--pelican-auth-cache-file",
        type=str,
        default=".pelican_auth_cache",
        help="Auth cache file for device flow (default: .pelican_auth_cache).",
    )
    parser.add_argument(
        "--pelican-storage-prefix",
        type=str,
        default="/icecube/wipac",
        help=(
            "Prefix present in pelican:// URI paths to strip when inferring the token scope path "
            "(default: /icecube/wipac)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training_checkpoint.json",
        help="Path to checkpoint file tracking processed files",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="model_checkpoint.pt",
        help="Path to model checkpoint file",
    )

    parser.add_argument(
        "--checkpoint-io",
        type=str,
        default="auto",
        choices=["auto", "local", "pelican"],
        help=(
            "Where to read/write --checkpoint and --model-checkpoint. "
            "auto: use PelicanFS only for pelican:// paths; local: always local disk; pelican: always PelicanFS (if configured)."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "rocm", "mps"],
        help="Torch device to use. 'auto' prefers cuda/rocm -> mps -> cpu.",
    )

    parser.add_argument(
        "--print-file-contents",
        action="store_true",
        help="Print a preview of each input file and exit (no training, no checkpoints).",
    )
    parser.add_argument(
        "--print-max-events",
        type=int,
        default=5,
        help="Max number of events to print per file when using --print-file-contents.",
    )
    
    # Model Hyperparameters
    parser.add_argument(
        "--cond-dim",
        type=int,
        default=4,
        help="Dimension of event conditions",
    )
    parser.add_argument(
        "--feat-dim",
        type=int,
        default=3,
        help="Dimension of muon features",
    )
    parser.add_argument(
        "--latent-dim-global",
        type=int,
        default=32,
        help="Global latent dimension",
    )
    parser.add_argument(
        "--latent-dim-local",
        type=int,
        default=16,
        help="Local latent dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--lambda-gp",
        type=float,
        default=10.0,
        help="Gradient penalty weight",
    )

    args = parser.parse_args()

    main(args)

