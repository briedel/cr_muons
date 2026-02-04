"""Pelican Federation utilities for distributed file access and token management."""

import fnmatch
import os
import posixpath
import random
import shutil
import sys
import tempfile
import threading
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse


def has_wildcards(s: str) -> bool:
    """Check if a string contains shell glob wildcards."""
    return any(ch in s for ch in ("*", "?", "["))


def infer_pelican_federation_url(pelican_uri: str) -> str | None:
    """Infer federation discovery URL (e.g. pelican://osg-htc.org) from a pelican URI."""
    try:
        u = urlparse(pelican_uri)
    except Exception:
        return None
    if u.scheme != "pelican" or not u.netloc:
        return None
    return f"pelican://{u.netloc}"


def pelican_uri_to_local_cache_path(pelican_uri: str, *, cache_dir: str) -> str:
    """Map a pelican:// URI to a stable local cache path.

    We preserve the full path under the cache directory to avoid collisions.
    Example:
      pelican://osg-htc.org/icecube/wipac/a/b/file.parquet
      -> <cache_dir>/icecube/wipac/a/b/file.parquet
    """
    u = urlparse(pelican_uri)
    rel = (u.path or "/").lstrip("/")
    return os.path.join(str(cache_dir), rel)


def retry_with_backoff(
    fn,
    *,
    what: str,
    attempts: int = 5,
    base_delay_s: float = 0.8,
    max_delay_s: float = 20.0,
):
    """Retry a callable with exponential backoff.

    Intended for flaky remote IO (Pelican director/data URLs). Prints a compact
    warning per retry.
    """
    attempts = max(1, int(attempts))
    base_delay_s = float(base_delay_s)
    max_delay_s = float(max_delay_s)

    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i >= attempts:
                break
            delay = min(max_delay_s, base_delay_s * (2 ** (i - 1)))
            # Add a bit of jitter to avoid thundering herds.
            delay *= (0.7 + 0.6 * random.random())
            print(
                f"Warning: {what} failed ({type(e).__name__}: {e}); "
                f"retry {i}/{attempts} in {delay:.1f}s"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"{what} failed after {attempts} attempts: {type(last_err).__name__}: {last_err}"
    ) from last_err


def prefetch_pelican_files(
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
        out_path = pelican_uri_to_local_cache_path(uri, cache_dir=cache_dir)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            local_paths.append(out_path)
            continue

        def _download_one() -> None:
            tmp_path = out_path + ".tmp"
            # Clean up any previous partial download.
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            print(f"Prefetching {uri} -> {out_path}")
            try:
                with fs.open(uri, "rb") as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                os.replace(tmp_path, out_path)
            finally:
                # If we failed before os.replace(), ensure tmp does not linger.
                try:
                    if os.path.exists(tmp_path) and (not os.path.exists(out_path)):
                        os.remove(tmp_path)
                except Exception:
                    pass

        retry_with_backoff(
            _download_one,
            what=f"prefetch download {uri}",
        )
        local_paths.append(out_path)

    return local_paths


class PelicanPrefetcher:
    """Background thread for prefetching Pelican files ahead of consumption."""

    def __init__(
        self,
        pelican_uris: list[str],
        *,
        federation_url: str,
        token: str | None,
        cache_dir: str,
        ahead: int,
        concurrency: int = 4,
        token_factory=None,
    ) -> None:
        self.pelican_uris = [str(u) for u in pelican_uris]
        self.uri_to_index = {u: i for i, u in enumerate(self.pelican_uris)}
        self.federation_url = federation_url
        self.token = token
        self.cache_dir = str(cache_dir)
        self.ahead = max(0, int(ahead))
        self.concurrency = max(1, int(concurrency))
        self.token_factory = token_factory
        self._token_lock = threading.Lock()

        self._current_index = -1
        self._current_index_shared = multiprocessing.Value('i', -1)
        self._next_download = 0
        self._stop = False
        self._status: dict[str, str] = {u: "pending" for u in self.pelican_uris}
        self._errors: dict[str, str] = {}

        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="pelican-prefetch", daemon=True)
        self._executor = None

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
                with self._current_index_shared.get_lock():
                    self._current_index_shared.value = idx
                self._cond.notify_all()

    def local_path(self, uri: str) -> str:
        return pelican_uri_to_local_cache_path(str(uri), cache_dir=self.cache_dir)

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

        self._executor = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="prefetch-worker")

        def _do_download(uri: str, out_path: str) -> None:
            def _download_one() -> None:
                # Check again if already done by another process (DDP)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return

                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                
                # Dynamic token resolution
                token_to_use = self.token
                if self.token_factory:
                    try:
                        with self._token_lock:
                            # This call should be fast (cached) in the helper usually
                            refreshed = self.token_factory()
                            if refreshed:
                                token_to_use = refreshed
                    except Exception as e:
                         # Log but continue with old token if possible
                        print(f"Warning: Prefetcher token refresh failed: {e}")

                headers = {"Authorization": f"Bearer {token_to_use}"} if token_to_use else {}

                tmp_path = f"{out_path}.{random.getrandbits(32)}.tmp"
                try:
                    # Create a fresh FS instance per download to avoid fsspec thread-safety issues
                    # (fixes 'ValueError: list.remove(x): x not in list' during concurrent reads)
                    fs_local = PelicanFileSystem(self.federation_url, headers=headers)
                    with fs_local.open(uri, "rb") as src, open(tmp_path, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                    os.replace(tmp_path, out_path)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except:
                            pass

            try:
                retry_with_backoff(_download_one, what=f"prefetch {uri}")
                with self._cond:
                    self._status[uri] = "done"
                    self._cond.notify_all()
            except Exception as e:
                with self._cond:
                    self._status[uri] = "error"
                    self._errors[uri] = str(e)
                    self._cond.notify_all()

        while True:
            with self._cond:
                # Update local index from shared value (updates from workers)
                with self._current_index_shared.get_lock():
                    if self._current_index_shared.value > self._current_index:
                        self._current_index = self._current_index_shared.value

                if self._stop:
                    if self._executor:
                        self._executor.shutdown(wait=False)
                    return

                target = min(len(self.pelican_uris), self._current_index + self.ahead + 1)
                
                # Check for files to dispatch
                while self._next_download < target:
                    uri = self.pelican_uris[self._next_download]
                    out_path = self.local_path(uri)
                    
                    # Skip if already being handled or done
                    if self._status[uri] != "pending":
                        self._next_download += 1
                        continue
                    
                    # Mark as downloading and dispatch
                    self._status[uri] = "downloading"
                    self._executor.submit(_do_download, uri, out_path)
                    self._next_download += 1
                
                # Wait for next update or finished download
                self._cond.wait(timeout=0.5)


def pelican_uri_dir_and_pattern(pelican_uri: str) -> tuple[str, str]:
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
        if item.startswith("pelican://") and has_wildcards(item):
            if inferred_fed is None:
                inferred_fed = infer_pelican_federation_url(item)
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
            has_recursive = "**" in item
            
            if callable(glob_fn):
                try:
                    # For pelicanfs, pass detail=False to get list of paths instead of dicts
                    # This significantly improves performance for large recursive globs with **
                    import inspect
                    sig = inspect.signature(glob_fn)
                    if 'detail' in sig.parameters:
                        res = glob_fn(item, detail=False)
                    else:
                        res = glob_fn(item)
                    if isinstance(res, (list, tuple)):
                        matches = [str(x) for x in res]
                except Exception as glob_err:
                    # If glob fails on a recursive pattern, raise immediately
                    if has_recursive:
                        raise RuntimeError(
                            f"Failed to expand recursive wildcard pattern '{item}'. "
                            f"Check:\n"
                            f"  - Federation URL is correct (--federation-url {inferred_fed})\n"
                            f"  - Base path exists in Pelican (try without **/*.parquet first)\n"
                            f"  - Token is valid and not expired (use --auto-token or --token)\n"
                            f"  - Director is reachable\n"
                            f"Error: {type(glob_err).__name__}: {glob_err}"
                        ) from glob_err
                    # For simple patterns, fall back to ls()
                    matches = []

            # Fallback: ls(dir) + fnmatch on basename (only for non-recursive patterns).
            if not matches and not has_recursive:
                dir_uri, basename_pat = pelican_uri_dir_and_pattern(item)
                ls_fn = getattr(fs, "ls", None)
                if not callable(ls_fn):
                    raise RuntimeError(
                        "Pelican filesystem does not support glob() or ls(); cannot expand wildcards."
                    )
                try:
                    entries = ls_fn(dir_uri)
                except Exception as e:
                    raise RuntimeError(
                        f"Pelican director failed to list directory '{dir_uri}'. "
                        f"Check:\n"
                        f"  - Federation URL is correct (--federation-url {inferred_fed})\n"
                        f"  - Path exists in Pelican\n"
                        f"  - Token is valid and not expired (use --auto-token or --token)\n"
                        f"  - Director is reachable\n"
                        f"Error: {type(e).__name__}: {e}"
                    ) from e
                if entries and isinstance(entries[0], dict):
                    paths = [str(e.get("name", "")) for e in entries if e.get("name")]
                else:
                    paths = [str(e) for e in (entries or [])]
                matches = [p for p in paths if fnmatch.fnmatch(posixpath.basename(p), basename_pat)]
            elif not matches and has_recursive:
                # Recursive pattern but no matches from glob()
                raise FileNotFoundError(
                    f"Recursive wildcard pattern matched 0 files: {item}\n"
                    f"Check:\n"
                    f"  - Base path exists: {item.split('**')[0]}\n"
                    f"  - Files match the pattern in subdirectories"
                )

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


def is_pelican_path(p: str) -> bool:
    """Check if a path is a pelican:// URI."""
    return str(p).startswith("pelican://")


def infer_scope_path_from_pelican_uri(pelican_uri: str, *, storage_prefix: str = "/icecube/wipac") -> str:
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

    # Strip ALL wildcards and range patterns from the path to find the base directory for token scope.
    # Token for a parent directory grants access to all subdirectories.
    # Find the first path component that contains wildcards or range patterns and use the parent directory.
    import re
    range_pattern = re.compile(r'^\d{7}-\d{7}$')  # Matches patterns like 0000000-0000999
    
    parts = path.split("/")
    clean_parts = []
    for part in parts:
        # Stop at the first component with any wildcard pattern or range directory
        if has_wildcards(part) or range_pattern.match(part):
            break
        clean_parts.append(part)
    
    # Reconstruct the clean path
    if clean_parts:
        path = "/".join(clean_parts)
    else:
        path = "/"
    
    # Normalize: remove trailing slashes except for root
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
        if not path:
            path = "/"

    if not path.startswith("/"):
        path = f"/{path}"
    return path


def fetch_pelican_token_via_helper(
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
    import importlib.util

    # Ensure device-flow instructions are visible.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        from utils.pelican.token_lib import get_access_token
    except ImportError:
        # Fallback: Load module directly from file path
        # Path(__file__).resolve().parents[2] goes up from src/utils/pelican_utils.py to repo root
        repo_root = Path(__file__).resolve().parents[2]
        token_lib_path = repo_root / "utils" / "pelican" / "token_lib.py"
        
        if not token_lib_path.exists():
            raise ImportError(
                f"Auto-token requires the local token helper library at {token_lib_path}. "
                "Make sure your environment includes the dependencies from requirements.txt."
            )
        
        spec = importlib.util.spec_from_file_location("token_lib", token_lib_path)
        token_lib = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(token_lib)
            get_access_token = token_lib.get_access_token
        except Exception as e:
            raise ImportError(
                f"Failed to load token_lib from {token_lib_path}: {e}"
            ) from e

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


def get_filesystem(federation_url, token):
    """Create a PelicanFileSystem instance or return None for local filesystem."""
    if federation_url:
        try:
            from pelicanfs.core import PelicanFileSystem
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            return PelicanFileSystem(federation_url, headers=headers)
        except ImportError:
            print("Warning: pelicanfs not found, falling back to local filesystem")
            return None
    return None


def select_checkpoint_fs(path: str | None, *, fs, mode: str) -> object | None:
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
    return fs if is_pelican_path(str(path)) else None
