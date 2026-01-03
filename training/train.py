import argparse
import json
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from pathlib import Path
from dataloader import SingleHDF5Dataset, MultiHDF5Dataset, ragged_collate_fn
from hf_dataloader import get_hf_dataset, hf_collate_fn
from normalizer import DataNormalizer
from model import ScalableGenerator, ScalableCritic, train_step_scalable

from torch.utils.data import DataLoader


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
            headers = {f"Authorization": f"Bearer {token}"} if token else None
            return PelicanFileSystem(federation_url, headers=headers)
        except ImportError:
            print("Warning: pelicanfs not found, falling back to local filesystem")
            return None
    return None

def load_progress(checkpoint_path, fs=None):
    processed_files = set()
    
    if fs:
        try:
            if fs.exists(checkpoint_path):
                with fs.open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    processed_files = set(data.get('processed_files', []))
        except Exception as e:
            print(f"Warning: Could not read checkpoint from Pelican: {e}")
    elif checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            try:
                data = json.load(f)
                processed_files = set(data.get('processed_files', []))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode checkpoint file {checkpoint_path}")
    return processed_files

def save_progress(checkpoint_path, processed_files, fs=None):
    if checkpoint_path:
        if fs:
            try:
                with fs.open(checkpoint_path, 'w') as f:
                    json.dump({'processed_files': list(processed_files)}, f)
            except Exception as e:
                print(f"Warning: Could not save checkpoint to Pelican: {e}")
        else:
            with open(checkpoint_path, 'w') as f:
                json.dump({'processed_files': list(processed_files)}, f)

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
            with fs.open(path, 'wb') as f:
                torch.save(checkpoint_data, f)
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
    # Initialize Filesystem for (optional) remote reads
    fs = get_filesystem(args.federation_url, args.token)

    # Print-only mode: preview file contents and exit
    if args.print_file_contents:
        for p in [str(x) for x in args.infiles]:
            _print_file_contents(p, fs=fs, max_events=args.print_max_events)
        return

    device = _select_torch_device(args.device)
    print(f"Using device: {device} ({_device_backend_label(device)})")

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
    
    opt_G = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_C = optim.Adam(crit.parameters(), lr=1e-4, betas=(0.0, 0.9))

    # Initialize Filesystem for Checkpoints
    # (already initialized above)

    # Load Checkpoints
    processed_files = load_progress(args.checkpoint, fs=fs)
    start_epoch = load_model_checkpoint(args.model_checkpoint, gen, crit, opt_G, opt_C, device, fs=fs)
    
    # Convert input paths to strings for consistent comparison
    all_files = [str(p) for p in args.infiles]
    files_to_process = [f for f in all_files if f not in processed_files]
    
    print(f"Total files: {len(all_files)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Remaining: {len(files_to_process)}")

    normalizer = DataNormalizer()

    file_pbar = tqdm(files_to_process, desc="Files", unit="file")
    for file_path in file_pbar:
        file_pbar.set_description(f"Processing {os.path.basename(file_path)}")
        
        if args.use_hf:
            # Infer format from file extension
            file_format = 'parquet' if file_path.endswith('.parquet') else 'h5'
            
            dataset = get_hf_dataset([file_path], 
                                     file_format=file_format,
                                     streaming=True,
                                     federation_url=args.federation_url,
                                     token=args.token)
            collate = hf_collate_fn
        else:
            # Assume HDF5 for non-HF path
            dataset = SingleHDF5Dataset(file_path)
            collate = ragged_collate_fn
        
        dataloader = DataLoader(dataset,
                                batch_size=4,
                                collate_fn=collate)

        batch_pbar = tqdm(dataloader, desc="Batches", leave=False)
        for real_muons, batch_idx, prims, counts in batch_pbar:
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

            

            # Move to device
            real_muons_feats = real_muons_feats.to(device)
            batch_idx = batch_idx.to(device)
            prims_feats = prims_feats.to(device)
            counts = counts.to(device)

            # Normalize
            real_muons_norm = normalizer.normalize_features(real_muons_feats)
            prims_norm = normalizer.normalize_primaries(prims_feats)
            
            c_loss, g_loss = train_step_scalable(
                gen, crit, opt_G, opt_C,
                real_muons_norm, batch_idx, prims_norm, counts,
                lambda_gp=args.lambda_gp,
                device=device
            )
            
            batch_pbar.set_postfix(c_loss=f"{c_loss:.4f}", g_loss=f"{g_loss:.4f}")

        # Checkpoint after file is done
        processed_files.add(file_path)
        save_progress(args.checkpoint, processed_files, fs=fs)
        save_model_checkpoint(args.model_checkpoint, gen, crit, opt_G, opt_C, fs=fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--infiles",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Use Hugging Face Streaming Dataset",
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

