import argparse
import os
import torch
import numpy as np
import pandas as pd
from src.datamodules.muon_datamodule import MuonDataModule
from src.utils.lazy_preprocessor import LazyDataPreprocessor
from src.utils.pelican_utils import (
    expand_pelican_wildcards,
    fetch_pelican_token_via_helper,
    infer_scope_path_from_pelican_uri,
    is_pelican_path,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fit and Test LazyDataPreprocessor")
    # Data args
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data")
    parser.add_argument("--federation_url", type=str, default=None, help="Pelican URL")
    parser.add_argument("--file_format", type=str, default="parquet", help="hdf5 or parquet")

    # Pelican / Authtokens
    parser.add_argument("--auto_token", action="store_true", help="Automatically fetch via osg-token-scope")
    parser.add_argument("--token", type=str, default=None, help="Manually provided token")
    parser.add_argument("--pelican_scope_path", type=str, default=None)
    parser.add_argument("--pelican_storage_prefix", type=str, default="/icecube/wipac")
    parser.add_argument("--pelican_oidc_url", type=str, default="https://token-issuer.icecube.aq")
    parser.add_argument("--pelican_auth_cache_file", type=str, default="pelican_auth_cache.json")
    
    # Preprocessor args
    parser.add_argument("--method", type=str, default="power", choices=['standard', 'power', 'quantile', 'minmax', 'robust'])
    parser.add_argument("--num_fit_batches", type=int, default=200, help="Batches to collect for fitting")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="preprocessor_test", help="Where to save plots and scaler")
    parser.add_argument("--save_scaler", action="store_true", help="Save the fitted scaler")
    parser.add_argument("--resume", action="store_true", help="Resume fitting from an existing scaler (only for methods supporting partial_fit)")

    # DataLoader Args
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--no_parquet_batch_reader", action="store_false", dest="parquet_batch_reader", help="Disable efficient Parquet batch reader")
    parser.set_defaults(parquet_batch_reader=True)
    parser.add_argument("--shuffle_parquet", action="store_true", help="Shuffle parquet files")
    parser.add_argument("--multi_file_shuffle", type=int, default=0, help="Number of files to interleave batches from")
    parser.add_argument("--prefetch_ahead", type=int, default=0, help="Number of Pelican files to prefetch ahead")
    parser.add_argument("--prefetch_concurrency", type=int, default=4, help="Number of concurrent Pelican downloads")
    parser.add_argument("--prefetch_dir", type=str, default="prefetch_cache", help="Cache directory for Pelican prefetching")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.prefetch_dir:
        os.makedirs(args.prefetch_dir, exist_ok=True)
        print(f"Prefetch cache directory: {os.path.abspath(args.prefetch_dir)}")
    
    # Pelican Token logic
    token = args.token
    token_scope_path = args.pelican_scope_path
    
    if args.auto_token: 
        if not token_scope_path and is_pelican_path(args.data_dir):
            token_scope_path = infer_scope_path_from_pelican_uri(
                args.data_dir, 
                storage_prefix=args.pelican_storage_prefix
            )
        
        if not token:
            print(f"Fetching auto-token for scope: {token_scope_path}")
            try:
                token = fetch_pelican_token_via_helper(
                    scope_path=token_scope_path,
                    federation_url=args.federation_url,
                    oidc_url=args.pelican_oidc_url,
                    auth_cache_file=args.pelican_auth_cache_file,
                    storage_prefix=args.pelican_storage_prefix,
                )
                print("Token acquired successfully.")
            except Exception as e:
                print(f"Failed to fetch token: {e}")

    token_refresh_args = None
    if args.auto_token and token_scope_path:
        token_refresh_args = {
            "scope_path": token_scope_path,
            "oidc_url": args.pelican_oidc_url,
            "auth_cache_file": args.pelican_auth_cache_file,
            "storage_prefix": args.pelican_storage_prefix
        }

    # 1. Setup Data & Fit Loop
    import time
    from src.utils.pelican_utils import expand_pelican_wildcards, prefetch_pelican_files, get_filesystem
    import random
    
    # Expand files once
    print(f"Expanding wildcards: {args.data_dir}")
    files, inferred_fed = expand_pelican_wildcards(
        [args.data_dir], 
        federation_url=args.federation_url, 
        token=token
    )
    if not args.federation_url and inferred_fed:
        args.federation_url = inferred_fed
    print(f"Found {len(files)} files.")
    
    if args.shuffle_parquet:
        random.shuffle(files)
        
    # Manual Chunking for download-process-delete cycle
    chunk_size = args.prefetch_ahead if args.prefetch_ahead > 0 else (args.multi_file_shuffle or 4)
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    
    # Initialize Preprocessor
    print(f"Initializing LazyDataPreprocessor (method={args.method})...")
    preprocessor = LazyDataPreprocessor(method=args.method)
    scaler_path = os.path.join(args.output_dir, f"muon_scaler_{args.method}.joblib")
    if args.resume and os.path.exists(scaler_path):
        preprocessor.load(scaler_path)
    
    # Initialize Filesystem for downloading
    fs = get_filesystem(args.federation_url, token)
    
    # Fit Loop
    total_processed_batches = 0
    
    # Check if method supports partial_fit
    supports_partial = hasattr(preprocessor, 'partial_fit') and args.method in ['standard', 'minmax', 'maxabs']
    if not supports_partial:
        print(f"\nNOTICE: Method '{args.method}' does not support incremental update (partial_fit).")
        print(f"Strategy: We will download a single chunk of files, fit on that representative sample, and then stop.")
        print(f"Ensure --prefetch_ahead (or chunk size) is large enough to capture the distribution (e.g. >20 files).")
    
    print(f"Starting Chunked Fit Loop (Goal: {args.num_fit_batches} batches)...")
    
    loader_last = None # Keep last loader for visualization
    last_chunk_files = [] 
    
    for i, chunk_files in enumerate(file_chunks):
        if total_processed_batches >= args.num_fit_batches:
            print("Reached target number of batches.")
            break
            
        print(f"\n--- Processing Chunk {i+1}/{len(file_chunks)} ({len(chunk_files)} files) ---")
        
        # 0. Clean previous chunk if different
        if last_chunk_files and args.prefetch_dir:
            for f in last_chunk_files:
                if os.path.exists(f): os.remove(f)
        
        # 1. Download Chunk
        t0 = time.time()
        local_files = []
        if is_pelican_path(chunk_files[0]):
             path_dir = args.prefetch_dir or "prefetch_cache"
             print(f"Downloading {len(chunk_files)} files to {path_dir}...")
             local_files = prefetch_pelican_files(chunk_files, fs=fs, cache_dir=path_dir)
        else:
             local_files = chunk_files
        print(f"Chunk downloaded in {time.time()-t0:.2f}s")
        last_chunk_files = local_files
             
        # 2. Setup DataModule for this chunk
        dm = MuonDataModule(
            data_dir=None,
            files_override=local_files,
            batch_size=2048, 
            file_format=args.file_format,
            num_workers=args.num_workers,
            parquet_batch_reader=args.parquet_batch_reader,
            muon_feature_selection="all",
            token=None, # Local files don't need token
            federation_url=None,
            shuffle_parquet=True, # Shuffle within the chunk
            prefetch_ahead=0, # Disable internal prefetcher
        )
        dm.setup()
        loader = dm.train_dataloader()
        loader_last = loader
        
        # 3. Fit
        t1 = time.time()
        if supports_partial:
            # Consume whole chunk to maximize efficiency
            count = preprocessor.partial_fit(loader, num_batches=999999, feature_extractor=lambda b: b[0])
            total_processed_batches += count
            print(f"Chunk fit complete: {count} batches processed (Total: {total_processed_batches}). Time: {time.time()-t1:.2f}s")
        else:
            # Non-partial methods (power, quantile)
            # We fit on this chunk entirely, then break because we can't update.
            print(f"Fitting global scaler ({args.method}) on this chunk...")
            preprocessor.fit(loader, num_batches=args.num_fit_batches, feature_extractor=lambda b: b[0])
            print(f"Fit complete on representative sample. Stopping.")
            break

    if args.save_scaler:
        preprocessor.save(scaler_path)

    # 3. Visualization
    print("Generating visualization...")
    
    # Collect a fresh batch for visualization
    batch = None
    if loader_last:
        try:
            # Create fresh iterator from the last chunk (files should still exist)
            iterator = iter(loader_last)
            batch = next(iterator)
        except StopIteration:
            print("Warning: Last chunk exhausted. Cannot visualize.")
    
    if batch is not None:
        raw_tensor = batch[0]
        
        # Transform
    transformed_tensor = preprocessor.transform(raw_tensor)
    
    # Inverse Transform Check
    reconstructed_tensor = preprocessor.inverse_transform(transformed_tensor)
    
    # Convert to Numpy
    raw = raw_tensor.detach().cpu().numpy()
    trans = transformed_tensor.detach().cpu().numpy()
    rec = reconstructed_tensor.detach().cpu().numpy()
    
    feature_count = raw.shape[1]
    
    # Plot Histograms
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(feature_count, 3, figsize=(15, 4 * feature_count))
    if feature_count == 1:
        axes = [axes]
        
    feature_names = [f"Feat {i}" for i in range(feature_count)]
    if feature_count >= 1: feature_names[0] = "Energy (GeV)"
    # Attempt to guess names if possible, but keeping generic is safer
    
    for i in range(feature_count):
        # Raw
        ax = axes[i][0]
        ax.hist(raw[:, i], bins=50, alpha=0.7, color='blue', label='Raw')
        ax.set_title(f"{feature_names[i]} - Raw")
        ax.legend()
        
        # Transformed
        ax = axes[i][1]
        ax.hist(trans[:, i], bins=50, alpha=0.7, color='green', label=f'Transformed ({args.method})')
        ax.set_title(f"{feature_names[i]} - Transformed\nRange: [{trans[:, i].min():.2f}, {trans[:, i].max():.2f}]")
        ax.legend()
        
        # Inverse Error
        ax = axes[i][2]
        error = np.abs(raw[:, i] - rec[:, i])
        ax.hist(error, bins=50, alpha=0.7, color='red', label='Reconstruction Error')
        ax.set_title(f"Inv. Transform Error (Max: {error.max():.2e})")
        ax.set_yscale('log')
        ax.legend()
        
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"transform_comparison_{args.method}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Print Stats
    print("\nTransformation Statistics:")
    print(f"{'Feature':<10} | {'Raw Min':<10} | {'Raw Max':<10} | {'Trans Min':<10} | {'Trans Max':<10} | {'Trans Mean':<10} | {'Trans Std':<10}")
    print("-" * 90)
    for i in range(feature_count):
        print(f"{i:<10} | {raw[:, i].min():<10.2e} | {raw[:, i].max():<10.2e} | "
              f"{trans[:, i].min():<10.2f} | {trans[:, i].max():<10.2f} | "
              f"{trans[:, i].mean():<10.2f} | {trans[:, i].std():<10.2f}")

if __name__ == "__main__":
    main()
