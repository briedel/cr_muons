import argparse
import json
import os
import posixpath
import subprocess
import sys
import shutil
import time
from urllib.parse import urlparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from collections import deque
import math

from utils import (
    expand_pelican_wildcards,
    device_backend_label,
    fetch_pelican_token_via_helper,
    first_batch_signature,
    fs_put_file,
    fs_put_json,
    get_filesystem,
    GPUUsageTracker,
    infer_file_format,
    infer_pelican_federation_url,
    infer_scope_path_from_pelican_uri,
    is_pelican_path,
    load_model_checkpoint,
    load_progress,
    OutlierParquetWriter,
    PelicanPrefetcher,
    pelican_uri_to_local_cache_path,
    PrefetchIterator,
    prefetch_pelican_files,
    print_file_contents,
    save_model_checkpoint,
    save_progress,
    select_checkpoint_fs,
    select_torch_device,
)

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
            if is_pelican_path(p):
                inferred_fed = infer_pelican_federation_url(p)
                break
    if args.federation_url is None and inferred_fed is not None:
        args.federation_url = inferred_fed

    # If pelican paths are present but no token flow is enabled, warn early.
    # Prefer checkpoint paths for scope inference (they may require write access).
    pelican_paths_all = [p for p in (checkpoint_paths + raw_infiles) if is_pelican_path(p)]
    pelican_checkpoint_paths = [p for p in checkpoint_paths if is_pelican_path(p)]
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
                scope_path = infer_scope_path_from_pelican_uri(
                    pelican_paths[0],
                    storage_prefix=args.pelican_storage_prefix,
                )

            print(f"Fetching Pelican token for scope: {scope_path}")
            args.token = fetch_pelican_token_via_helper(
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
    checkpoint_fs = select_checkpoint_fs(args.checkpoint, fs=fs, mode=getattr(args, "checkpoint_io", "auto"))
    model_checkpoint_fs = select_checkpoint_fs(
        args.model_checkpoint,
        fs=fs,
        mode=getattr(args, "checkpoint_io", "auto"),
    )

    # Print-only mode: preview file contents and exit
    if args.print_file_contents:
        for p in [str(x) for x in args.infiles]:
            print_file_contents(p, fs=fs, max_events=args.print_max_events)
        return

    device = select_torch_device(args.device)
    print(f"Using device: {device} ({device_backend_label(device)})")

    # Optional: enable TF32 on supported CUDA hardware, with clear warnings otherwise
    tf32_enabled = False
    if bool(getattr(args, "allow_tf32", False)):
        if not str(device).startswith("cuda"):
            print("Warning: --allow-tf32 was set but device is not CUDA; ignoring.")
        else:
            major = minor = 0
            try:
                dev_index = torch.cuda.current_device()
                major, minor = torch.cuda.get_device_capability(dev_index)
            except Exception:
                pass
            if major < 8:
                print(f"Warning: --allow-tf32 was set but GPU sm{major}{minor} does not support TF32; ignoring.")
            else:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")
                    tf32_enabled = True
                    print("TF32 enabled for CUDA matmuls/cuDNN")
                except Exception:
                    print("Warning: failed to enable TF32; continuing without it.")

    # Start background GPU usage tracker for time-averaged reporting
    gpu_tracker = GPUUsageTracker(device, sample_interval=0.1, window_size=600)
    gpu_tracker.start()

    # Print GP tuning info if enabled
    lambda_gp = float(getattr(args, "lambda_gp", 0.0) or 0.0)
    if lambda_gp > 0:
        gp_max = int(getattr(args, "gp_max_pairs", 0) or 0)
        gp_frac = float(getattr(args, "gp_sample_fraction", 0.0) or 0.0)
        gp_every = int(getattr(args, "gp_every", 1) or 1)
        gp_info_parts = [f"λ={lambda_gp}"]
        if gp_max > 0:
            gp_info_parts.append(f"max_pairs={gp_max}")
        if gp_frac > 0.0:
            gp_info_parts.append(f"sample_frac={gp_frac:.2f}")
        if gp_every > 1:
            gp_info_parts.append(f"every={gp_every}")
        print(f"Gradient penalty: {', '.join(gp_info_parts)}")
    
    # Print config summary for easy reference
    pooling_mode = str(getattr(args, "critic_pooling", "amax") or "amax").lower()
    print(f"Config: critic_pooling={pooling_mode}, tf32={'enabled' if tf32_enabled else 'disabled'}")

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
        # Log configuration flags for easy comparison in TensorBoard
        try:
            pooling_mode = str(getattr(args, "critic_pooling", "amax") or "amax").lower()
            writer.add_text("config/critic_pooling", pooling_mode, 0)
            writer.add_scalar("config/critic_pooling_mean", 1.0 if pooling_mode == "mean" else 0.0, 0)
            writer.add_scalar("cuda/tf32_enabled", 1.0 if tf32_enabled else 0.0, 0)
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

        tb_sync_fs = select_checkpoint_fs(tb_sync_to, fs=fs, mode=getattr(args, "tb_io", "auto"))
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
                        fs_put_file(tb_sync_fs, dest_path, local_path)
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
        device=device,
        pooling_mode=str(getattr(args, "critic_pooling", "amax") or "amax")
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
    
    # Instantiate optimizers
    optimizer_type = str(getattr(args, "optimizer", "adam") or "adam").lower()
    lr = float(getattr(args, "lr", 1e-4) or 1e-4)
    
    if optimizer_type == "sgd":
        momentum = float(getattr(args, "momentum", 0.9) or 0.9)
        opt_G = optim.SGD(gen.parameters(), lr=lr, momentum=momentum, dampening=0, nesterov=False)
        opt_C = optim.SGD(crit.parameters(), lr=lr, momentum=momentum, dampening=0, nesterov=False)
        print(f"Using SGD optimizer: lr={lr}, momentum={momentum}")
    elif optimizer_type == "adam":
        opt_G = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
        opt_C = optim.Adam(crit.parameters(), lr=lr, betas=(0.0, 0.9))
        print(f"Using Adam optimizer: lr={lr}")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Choose 'adam' or 'sgd'.")

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
    prefetcher: PelicanPrefetcher | None = None
    if getattr(args, "prefetch_dir", None):
        pelican_candidates = [str(p) for p in files_to_process if is_pelican_path(p)]
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
                prefetcher = PelicanPrefetcher(
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
                prefetch_pelican_files(
                    pelican_candidates,
                    fs=fs,
                    cache_dir=args.prefetch_dir,
                )
                print(f"Prefetch complete. Training will read from: {args.prefetch_dir}")

    normalizer = DataNormalizer()

    outlier_writer = None
    outliers_dir = getattr(args, "outliers_dir", None)
    if outliers_dir:
        outlier_writer = OutlierParquetWriter(str(outliers_dir))
        ev_thr = int(getattr(args, "max_muons_per_event", 0) or 0)
        thr_label = "--max-muons-per-event" if ev_thr > 0 else "c"
        thr_val = ev_thr if ev_thr > 0 else int(getattr(args, "max_muons_per_batch", 0) or 0)
        print(f"Outlier capture enabled: {outliers_dir} (threshold {thr_label}={thr_val})")

    memory_cache = None
    mem_cache_mb = int(getattr(args, "memory_cache_mb", 0) or 0)
    if mem_cache_mb > 0:
        memory_cache = FileBytesLRUCache(max_bytes=mem_cache_mb * 1024 * 1024)
        print(f"In-memory parquet cache enabled: {mem_cache_mb} MiB (process-local)")

    train_steps_done = 0
    # Track 500-step moving average of Wasserstein gap
    w_gap_window = 500
    w_gap_hist: deque[float] = deque(maxlen=w_gap_window)
    w_gap_sum = 0.0
    first_batch_printed = False  # Track if we've printed the first batch
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
        if getattr(args, "prefetch_dir", None) and is_pelican_path(file_path):
            cached = pelican_uri_to_local_cache_path(file_path, cache_dir=args.prefetch_dir)
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
        if not is_pelican_path(read_path) and not os.path.exists(read_path):
            raise FileNotFoundError(
                f"Input file not found: {read_path}\n"
                "If this is a host filesystem path (e.g. /icecube/...), make sure it is available/mounted in your environment.\n"
                "If you intended to read via Pelican, pass a pelican:// URI (and optionally --federation-url/--token)."
            )

        file_format = infer_file_format(read_path)

        fed_for_read = args.federation_url if is_pelican_path(read_path) else None
        token_for_read = args.token if is_pelican_path(read_path) else None

        # Parquet and pelican:// inputs require the HF streaming loader in this repo.
        use_hf_for_file = bool(args.use_hf) or file_format == "parquet" or is_pelican_path(read_path)
        if (not args.use_hf) and use_hf_for_file and (file_format == "parquet" or is_pelican_path(read_path)):
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
            data_iter = PrefetchIterator(data_iter, max_prefetch=batch_prefetch)
        batch_pbar = tqdm(desc="Batches", unit="batch", leave=False)
        batches_seen = 0
        events_seen = 0
        muons_seen = 0
        skipped_empty = 0
        reported_first_batch = False

        profile_steps = int(getattr(args, "profile_steps", 0) or 0)
        steps_profiled = 0

        file_t0 = time.perf_counter()
        load_time_s = 0.0
        step_time_s = 0.0
        skipped_oversize = 0
        written_oversize = 0

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

            # Batch GPU→CPU transfers: pull counts once, reuse throughout batch processing
            counts_cpu = counts.detach().cpu()
            counts_sum_val = int(counts_cpu.sum().item())
            counts_numel_val = int(counts.numel())
            
            events_seen += counts_numel_val
            muons_seen += counts_sum_val
            # Preserve raw tensors for optional outlier capture.
            prims_raw = prims
            real_muons_raw = real_muons

            

            t_step0 = time.perf_counter()

            # Move to device. With pinned memory (see --pin-memory) these can be non-blocking.
            non_blocking = bool(getattr(args, "pin_memory", False)) and str(device).startswith("cuda")

            # Ragged batches can have huge variation in total muons. A rare outlier batch
            # can spike peak memory and cause PyTorch to reserve more GPU memory, which then
            # stays reserved (allocator cache). To prevent long-run OOM from repeated outliers,
            # optionally split a batch into microbatches bounded by a max total muon count.
            max_muons = int(getattr(args, "max_muons_per_batch", 0) or 0)
            max_event_muons = int(getattr(args, "max_muons_per_event", 0) or 0)
            event_muon_limit = max_event_muons if max_event_muons > 0 else max_muons

            # Per-event outlier handling should apply even when the *batch total* is not large.
            # If any event has count > event_muon_limit, optionally write it to --outliers-dir
            # and remove it from this batch so we can keep training on the rest safely.
            if event_muon_limit > 0:
                try:
                    oversize_mask = counts > int(event_muon_limit)
                    if bool(oversize_mask.any().item()):
                        oversize_idx = torch.nonzero(oversize_mask, as_tuple=False).flatten().tolist()

                        # Write outliers (raw columns, before ID stripping).
                        if outlier_writer is not None:
                            counts_for_outlier = counts_cpu.tolist()  # Reuse CPU-side counts
                            for ev in oversize_idx:
                                try:
                                    c = int(counts_for_outlier[ev])
                                    m_evt = (batch_idx == int(ev))
                                    mu_evt = real_muons_raw[m_evt]
                                    _ = outlier_writer.write_event(
                                        source_file=str(file_path),
                                        source_file_index=int(file_idx),
                                        batch_index=int(batches_seen),
                                        event_index=int(ev),
                                        count=int(c),
                                        primaries=prims_raw[int(ev)],
                                        muons=mu_evt,
                                    )
                                    written_oversize += 1
                                except Exception as e:
                                    tqdm.write(
                                        f"[file {file_idx}/{len(files_to_process)}] warning: failed to write outlier event (count={c}): {e}"
                                    )

                        skipped_oversize += len(oversize_idx)

                        # Filter out oversize events from primaries/counts and remap muons/batch_idx.
                        keep_mask = ~oversize_mask
                        keep_events = torch.nonzero(keep_mask, as_tuple=False).flatten()
                        if int(keep_events.numel()) == 0:
                            skipped_empty += 1
                            continue

                        # Map old event indices -> new compact indices.
                        # Example: keep_events=[0,2,5] => mapping {0:0,2:1,5:2}
                        old_to_new = torch.full((int(counts.numel()),), -1, dtype=torch.long)
                        old_to_new[keep_events] = torch.arange(int(keep_events.numel()), dtype=torch.long)

                        mu_keep = keep_mask[batch_idx]
                        real_muons_raw = real_muons_raw[mu_keep]
                        batch_idx = old_to_new[batch_idx[mu_keep]]
                        prims_raw = prims_raw[keep_mask]
                        counts = counts[keep_mask]
                        counts_cpu = counts.detach().cpu()  # Re-pull counts after filtering
                except Exception:
                    # If anything goes wrong, fall back to the original batch.
                    pass

            # Now handle IDs if present (from HF dataloader/Parquet)
            # Primaries: [Batch, 6] -> [Batch, 4] (Skip first 2)
            prims = prims_raw
            real_muons = real_muons_raw
            if prims.shape[1] == 6:
                prims_feats = prims[:, 2:]
            else:
                prims_feats = prims

            # Muons: [Total, 5] -> [Total, 3] (Skip first 2)
            if real_muons.shape[1] == 5:
                real_muons_feats = real_muons[:, 2:]
            else:
                real_muons_feats = real_muons

            # Optional: report a small signature so you can confirm data changes across files.
            # Must run after *_feats are defined.
            if (not reported_first_batch) and bool(getattr(args, "report_first_batch", False)):
                if real_muons_feats.numel() > 0 and counts_sum_val > 0:
                    sig = first_batch_signature(prims_feats, real_muons_feats, counts)
                    c_preview = counts_cpu[:8].tolist()
                    tqdm.write(
                        f"[file {file_idx}/{len(files_to_process)}] first_batch signature={sig} "
                        f"events={counts_numel_val} muons={counts_sum_val} counts[:8]={c_preview}"
                    )
                    reported_first_batch = True

            # Some events (or entire batches) can have zero muons. WGAN training
            # requires at least one real sample to compute losses and gradient penalty.
            if real_muons_feats.numel() == 0 or counts_sum_val == 0:
                skipped_empty += 1
                continue

            # Keep last-step tensors for optional TensorBoard histogram logging.
            real_muons_norm = None
            prims_norm = None
            counts_dev = None

            def _run_substep(
                sub_muons_feats: torch.Tensor,
                sub_batch_idx: torch.Tensor,
                sub_prims_feats: torch.Tensor,
                sub_counts: torch.Tensor,
            ):
                nonlocal steps_profiled
                
                sub_muons_feats = sub_muons_feats.to(device, non_blocking=non_blocking)
                sub_batch_idx = sub_batch_idx.to(device, non_blocking=non_blocking)
                sub_prims_feats = sub_prims_feats.to(device, non_blocking=non_blocking)
                sub_counts = sub_counts.to(device, non_blocking=non_blocking)

                # Normalize
                sub_muons_norm = normalizer.normalize_features(sub_muons_feats)
                sub_prims_norm = normalizer.normalize_primaries(sub_prims_feats)

                # Print first 10 samples (normalized vs unnormalized) for debugging
                nonlocal first_batch_printed
                if (not first_batch_printed) and (sub_muons_feats.numel() > 0):
                    first_batch_printed = True
                    n_show = min(10, int(sub_muons_feats.shape[0]))
                    tqdm.write("\n" + "="*80)
                    tqdm.write("[First Training Batch Sample Inspector]")
                    tqdm.write("="*80)
                    
                    # Muons: show feature dimensions
                    tqdm.write(f"\n--- Muon Features (first {n_show} samples) ---")
                    tqdm.write("Unnormalized:")
                    for i in range(n_show):
                        vals_str = "  ".join([f"{v:.6f}" for v in sub_muons_feats[i].cpu().numpy()])
                        tqdm.write(f"  [{i}] {vals_str}")
                    tqdm.write("\nNormalized:")
                    for i in range(n_show):
                        vals_str = "  ".join([f"{v:.6f}" for v in sub_muons_norm[i].detach().cpu().numpy()])
                        tqdm.write(f"  [{i}] {vals_str}")
                    
                    # Primaries: show all available events (up to 10)
                    n_prims_show = min(10, int(sub_prims_feats.shape[0]))
                    tqdm.write(f"\n--- Primary Features (first {n_prims_show} events) ---")
                    tqdm.write("Unnormalized:")
                    for i in range(n_prims_show):
                        vals_str = "  ".join([f"{v:.6f}" for v in sub_prims_feats[i].cpu().numpy()])
                        tqdm.write(f"  [{i}] {vals_str}")
                    tqdm.write("\nNormalized:")
                    for i in range(n_prims_show):
                        vals_str = "  ".join([f"{v:.6f}" for v in sub_prims_norm[i].detach().cpu().numpy()])
                        tqdm.write(f"  [{i}] {vals_str}")
                    tqdm.write("="*80 + "\n")

                # Profiling hook (optional)
                should_profile = (profile_steps > 0 and steps_profiled < profile_steps)
                if should_profile:
                    prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                    )
                    prof.__enter__()

                c_loss_local, g_loss_local, m_loss_local, w_gap_local = train_step_scalable(
                    gen,
                    crit,
                    opt_G,
                    opt_C,
                    sub_muons_norm,
                    sub_batch_idx,
                    sub_prims_norm,
                    sub_counts,
                    lambda_gp=args.lambda_gp,
                    critic_steps=int(getattr(args, "critic_steps", 1) or 1),
                    gp_max_pairs=int(getattr(args, "gp_max_pairs", 0) or 0),
                    gp_sample_fraction=float(getattr(args, "gp_sample_fraction", 0.0) or 0.0),
                    gp_every=int(getattr(args, "gp_every", 1) or 1),
                    grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0) or 0.0),
                    device=device,
                )

                if should_profile:
                    prof.__exit__(None, None, None)
                    try:
                        tqdm.write(f"[Profile step {steps_profiled+1}/{profile_steps}]")
                        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
                        tqdm.write(table)
                    except Exception as e:
                        tqdm.write(f"[Profile error: {e}]")
                    steps_profiled += 1

                return c_loss_local, g_loss_local, m_loss_local, w_gap_local, sub_muons_norm, sub_prims_norm, sub_counts

            # Default: single step on the full batch.
            updated_w_ma = False  # Track if w_gap moving average updated in branch
            if max_muons <= 0 or counts_sum_val <= max_muons:
                c_loss, g_loss, m_loss, w_gap, real_muons_norm, prims_norm, counts_dev = _run_substep(
                    real_muons_feats, batch_idx, prims_feats, counts
                )
                train_steps_done += 1
            else:
                # Microbatch by contiguous event ranges so batch_idx slicing is cheap.
                bsz = counts_numel_val
                start_ev = 0
                last_c = None
                last_g = None
                last_m = None
                last_w = None
                last_real_norm = None
                last_prims_norm = None
                last_counts_dev = None
                # Batch transfer of counts to CPU for fast loop iteration
                counts_cpu_list = counts_cpu.tolist()
                while start_ev < bsz:
                    # Choose an end_ev so sum(counts[start_ev:end_ev]) <= max_muons, at least 1 event.
                    cum = 0
                    end_ev = start_ev
                    while end_ev < bsz:
                        c = counts_cpu_list[end_ev]
                        # Corner case: a single event can exceed max_muons. In that case,
                        # microbatching cannot bound memory, so we skip the event to avoid
                        # catastrophic allocation spikes that can ratchet reserved memory upward.
                        if event_muon_limit > 0 and end_ev == start_ev and c > event_muon_limit:
                            # Oversize single-event outliers should have been filtered out earlier.
                            # Treat any remaining case as empty/skip to be safe.
                            skipped_oversize += 1
                            end_ev = start_ev + 1
                            cum = 0
                            break
                        if (end_ev > start_ev) and (cum + c > max_muons):
                            break
                        cum += c
                        end_ev += 1
                        if cum >= max_muons:
                            break
                    if end_ev <= start_ev:
                        end_ev = start_ev + 1

                    sub_counts = counts[start_ev:end_ev]
                    sub_counts_cpu = counts_cpu[start_ev:end_ev]
                    sub_counts_max = int(sub_counts_cpu.max().item())
                    if event_muon_limit > 0 and sub_counts_max > event_muon_limit:
                        # We already counted this as oversize; treat as skipped.
                        skipped_empty += 1
                        start_ev = end_ev
                        continue
                    # Skip all-zero microbatches (can happen with many 0-muon events).
                    sub_counts_sum = int(sub_counts_cpu.sum().item())
                    if sub_counts_sum > 0:
                        sub_prims = prims_feats[start_ev:end_ev]
                        m = (batch_idx >= start_ev) & (batch_idx < end_ev)
                        sub_muons = real_muons_feats[m]
                        sub_bidx = batch_idx[m] - start_ev
                        if sub_muons.numel() > 0:
                            last_c, last_g, last_m, last_w, last_real_norm, last_prims_norm, last_counts_dev = _run_substep(
                                sub_muons, sub_bidx, sub_prims, sub_counts
                            )
                            train_steps_done += 1
                        else:
                            skipped_empty += 1
                    else:
                        skipped_empty += 1

                    start_ev = end_ev

                # For logging, use the last microbatch losses.
                if last_c is None or last_g is None:
                    skipped_empty += 1
                    continue
                c_loss, g_loss = float(last_c), float(last_g)
                w_gap = float(last_w) if last_w is not None else 0.0
                # Update moving average buffer within microbatch path
                try:
                    if len(w_gap_hist) == w_gap_window:
                        w_gap_sum -= float(w_gap_hist[0])
                    w_gap_hist.append(float(w_gap))
                    w_gap_sum += float(w_gap)
                    updated_w_ma = True
                except Exception:
                    updated_w_ma = False
                real_muons_norm = last_real_norm
                prims_norm = last_prims_norm
                counts_dev = last_counts_dev

            # If not updated yet (e.g., full-batch path), update moving average now
            if not updated_w_ma:
                try:
                    if len(w_gap_hist) == w_gap_window:
                        w_gap_sum -= float(w_gap_hist[0])
                    w_gap_hist.append(float(w_gap))
                    w_gap_sum += float(w_gap)
                except Exception:
                    pass
            w_gap_ma = (w_gap_sum / max(1, len(w_gap_hist)))

            t_step1 = time.perf_counter()
            step_time_s += (t_step1 - t_step0)

            batch_pbar.set_postfix(c_loss=f"{c_loss:.4f}", g_loss=f"{g_loss:.4f}", m_loss=f"{m_loss:.4f}", w_gap=f"{w_gap:.4f}", w_ma=f"{w_gap_ma:.4f}")

            # TensorBoard logging (optional)
            if writer is not None:
                tb_every = int(getattr(args, "tb_log_interval", 0) or 0)
                if tb_every > 0 and (train_steps_done % tb_every == 0):
                    elapsed = max(1e-9, time.perf_counter() - file_t0)
                    avg_load_ms = (load_time_s / max(1, batches_seen)) * 1e3
                    avg_step_ms = (step_time_s / max(1, batches_seen)) * 1e3
                    writer.add_scalar("train/c_loss", float(c_loss), train_steps_done)
                    writer.add_scalar("train/g_loss", float(g_loss), train_steps_done)
                    writer.add_scalar("train/m_loss", float(m_loss), train_steps_done)
                    writer.add_scalar("train/w_gap", float(w_gap), train_steps_done)
                    try:
                        writer.add_scalar("train/w_gap_ma_500", float(w_gap_ma), train_steps_done)
                    except Exception:
                        pass
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
                        # Batch .item() calls: pull stats once
                        mean_counts_val = float(counts_cpu.float().mean().item())
                        max_counts_val = float(counts_cpu.max().item())
                        writer.add_scalar("data/mean_counts", mean_counts_val, train_steps_done)
                        writer.add_scalar("data/max_counts", max_counts_val, train_steps_done)
                    except Exception:
                        pass

                    mem = gpu_tracker.get_averaged_stats()
                    if mem is not None:
                        writer.add_scalar("cuda/alloc_mib", float(mem["alloc_mib"]), train_steps_done)
                        writer.add_scalar("cuda/reserved_mib", float(mem["reserved_mib"]), train_steps_done)
                        writer.add_scalar("cuda/max_alloc_mib", float(mem["max_alloc_mib"]), train_steps_done)
                        writer.add_scalar("cuda/max_reserved_mib", float(mem["max_reserved_mib"]), train_steps_done)
                        try:
                            writer.add_scalar("cuda/active_mib", float(mem["active_mib"]), train_steps_done)
                            writer.add_scalar("cuda/inactive_split_mib", float(mem["inactive_split_mib"]), train_steps_done)
                            writer.add_scalar("cuda/free_mib", float(mem["free_mib"]), train_steps_done)
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
                        if counts_dev is not None:
                            writer.add_histogram("data/counts", counts_dev.detach().to("cpu"), train_steps_done)
                    except Exception:
                        pass
                    try:
                        if prims_norm is None:
                            raise RuntimeError("prims_norm unavailable")
                        p_cpu = prims_norm.detach().to("cpu")
                        for d in range(min(int(p_cpu.shape[1]), 16)):
                            writer.add_histogram(f"data/primaries_dim{d}", p_cpu[:, d], train_steps_done)
                    except Exception:
                        pass

                    # Real vs fake muon feature histograms (normalized space)
                    try:
                        if real_muons_norm is None:
                            raise RuntimeError("real_muons_norm unavailable")
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
                            if prims_norm is None or counts_dev is None:
                                raise RuntimeError("prims_norm/counts unavailable")
                            fake_mu, _ = gen(prims_norm, counts_dev)
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
                mem = gpu_tracker.get_averaged_stats()
                mem_s = ""
                if mem is not None:
                    try:
                        dev_idx = int(mem.get("device_idx"))
                    except Exception:
                        dev_idx = -1
                    free = mem.get("free_mib", float("nan"))
                    active = mem.get("active_mib", float("nan"))
                    inact = mem.get("inactive_split_mib", float("nan"))
                    total = mem.get("total_mib", float("nan"))
                    alloc = mem.get("alloc_mib", 0)
                    usage_pct = float("nan")
                    try:
                        total_f = float(total)
                        alloc_f = float(alloc)
                        if total_f == total_f and total_f > 0:
                            usage_pct = (alloc_f / total_f) * 100.0
                    except Exception:
                        pass
                    mem_s = (
                        f" cuda:{dev_idx} alloc={mem['alloc_mib']:.0f}MiB res={mem['reserved_mib']:.0f}MiB"
                        f" max_alloc={mem['max_alloc_mib']:.0f}MiB"
                        f" active={active:.0f}MiB inact_split={inact:.0f}MiB free={free:.0f}MiB"
                        f" usage={usage_pct:.1f}%"
                    )
                # Loss health check: warn if out of expected ranges or non-finite
                warnings = []
                try:
                    if not math.isfinite(float(c_loss)):
                        warnings.append("c_loss non-finite")
                    if not math.isfinite(float(g_loss)):
                        warnings.append("g_loss non-finite")
                    if not math.isfinite(float(m_loss)):
                        warnings.append("m_loss non-finite")
                except Exception:
                    warnings.append("loss finite-check failed")

                # Heuristic ranges based on observed stable training
                # c_loss typically ~ [-300, 0]; g_loss ~ around 1000 due to clamp; m_loss < 0.2
                try:
                    if abs(float(c_loss)) > 1000.0:
                        warnings.append(f"c_loss out-of-range ({float(c_loss):.4f})")
                    g_val = float(g_loss)
                    if g_val < 100.0 or g_val > 5000.0:
                        warnings.append(f"g_loss out-of-range ({g_val:.4f})")
                    if float(m_loss) > 0.20:
                        warnings.append(f"m_loss high ({float(m_loss):.4f})")
                except Exception:
                    pass

                if warnings:
                    tqdm.write(
                        f"[file {file_idx}/{len(files_to_process)}] WARNING: " + "; ".join(warnings)
                    )
                    # Log to TensorBoard when loss warnings occur
                    if writer is not None:
                        try:
                            writer.add_scalar("status/loss_warning", 1.0, train_steps_done)
                            writer.add_text("status/loss_warning_msg", "; ".join(warnings), train_steps_done)
                        except Exception:
                            pass
                tqdm.write(
                    f"[file {file_idx}/{len(files_to_process)}] batches={batches_seen} "
                    f"events={events_seen} muons={muons_seen} "
                    f"skipped_empty={skipped_empty} skipped_oversize={skipped_oversize} written_oversize={written_oversize} "
                    f"rate: {bps:.2f} batch/s {eps:.1f} evt/s {mps:.1f} mu/s "
                    f"avg: load={avg_load_ms:.1f}ms step={avg_step_ms:.1f}ms "
                    f"c_loss={c_loss:.4f} g_loss={g_loss:.4f} m_loss={m_loss:.4f} w_gap={w_gap:.4f} w_ma={w_gap_ma:.4f}"
                    f"{mem_s}"
                )

        batch_pbar.close()

        # Explicitly clean up DataLoader and dataset to release file descriptors
        # This prevents "too many open files" errors when processing many files
        try:
            del data_iter
        except Exception:
            pass
        try:
            del dataloader
        except Exception:
            pass
        try:
            del dataset
        except Exception:
            pass
        
        # Force garbage collection to release resources immediately
        import gc
        gc.collect()

        # Checkpoint after file is done
        processed_files.add(file_path)
        # This script currently runs a single pass over files. We store epoch=0
        # for forward compatibility if you later add an epoch loop.
        save_progress(args.checkpoint, progress_epoch, processed_files, fs=checkpoint_fs)
        save_model_checkpoint(args.model_checkpoint, gen, crit, opt_G, opt_C, fs=model_checkpoint_fs)

        tqdm.write(
            f"[file {file_idx}/{len(files_to_process)}] done batches={batches_seen} events={events_seen} muons={muons_seen} skipped_empty={skipped_empty} skipped_oversize={skipped_oversize} written_oversize={written_oversize}"
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
    
    # Stop the GPU usage tracker
    gpu_tracker.stop()


def build_arg_parser() -> argparse.ArgumentParser:
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
        "--max-muons-per-batch",
        type=int,
        default=0,
        help=(
            "If >0, split each ragged batch into microbatches so that sum(counts) <= this value. "
            "Helps prevent GPU OOM when rare batches have extremely many muons (default: 0 disables)."
        ),
    )
    parser.add_argument(
        "--max-muons-per-event",
        type=int,
        default=0,
        help=(
            "If >0, treat any single event with count > this value as an outlier and skip it (and optionally write it via --outliers-dir). "
            "If 0, the outlier threshold falls back to --max-muons-per-batch. Default: 0."
        ),
    )
    parser.add_argument(
        "--outliers-dir",
        type=str,
        default=None,
        help=(
            "If set, write any single-event outliers (count > threshold) to this directory as Parquet "
            "(one small Parquet file per outlier event), and skip them during the main run. "
            "The threshold is --max-muons-per-event when set (>0), otherwise it falls back to --max-muons-per-batch."
        ),
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
        "--critic-step",
        dest="critic_steps",
        type=int,
        default=None,
        help="Alias for --critic-steps (kept for backward compatibility).",
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
        "--allow-tf32",
        action="store_true",
        help=(
            "Enable TensorFloat-32 for CUDA matmuls/cuDNN (Ampere+). Can improve GEMM/conv throughput with minor precision tradeoffs."
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
        "--critic-pooling",
        type=str,
        default="amax",
        choices=["amax", "mean"],
        help=(
            "Pooling mode for critic event aggregation: 'amax' (default) or 'mean'. 'mean' is cheaper to backpropagate."
        ),
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
    
    # Optimizer Hyperparameters
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help=(
            "Optimizer to use: 'adam' (default) or 'sgd'. "
            "SGD is faster (fewer GPU ops per step) but may require careful LR tuning."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer (default: 0.9, only used when --optimizer sgd).",
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
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help=(
            "If >0, apply gradient clipping (L2 norm) to generator and critic, including multiplicity step. "
            "Helps stabilize SGD and prevent NaNs. Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--gp-max-pairs",
        type=int,
        default=4096,
        help=(
            "Cap the number of interpolated muon pairs used for gradient penalty per step (default: 4096). "
            "Set 0 to disable capping."
        ),
    )
    parser.add_argument(
        "--gp-sample-fraction",
        type=float,
        default=0.0,
        help=(
            "If >0 and <1, randomly sample this fraction of aligned real/fake muon pairs for the gradient penalty. "
            "Applied after --gp-max-pairs (default: 0.0 disables)."
        ),
    )
    parser.add_argument(
        "--gp-every",
        type=int,
        default=2,
        help=(
            "Apply gradient penalty every N critic steps (default: 2). Set to 1 for every step."
        ),
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help=(
            "If >0, profile the first N training steps using torch.profiler and print timing breakdown. "
            "Useful for identifying bottlenecks (default: 0 disables)."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)

