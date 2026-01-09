"""Debugging and inspection utilities for data files."""

import hashlib

import torch


def infer_file_format(file_path: str) -> str:
    """Infer file format from extension."""
    p = str(file_path).lower()
    if p.endswith(".parquet") or p.endswith(".pq"):
        return "parquet"
    if p.endswith(".h5") or p.endswith(".hdf5"):
        return "h5"
    # Default to HDF5 for backward compatibility.
    return "h5"


def first_batch_signature(
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


def print_file_contents(file_path: str, fs=None, max_events: int = 5):
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
