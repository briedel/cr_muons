import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .hdf5_dataset import MultiHDF5Dataset, ragged_collate_fn
from .hf_dataset import get_hf_dataset, hf_collate_fn
from .parquet_dataset import get_parquet_batch_dataset
from ..utils.pelican_utils import expand_pelican_wildcards, PelicanPrefetcher
from ..utils.data_utils import PrefetchIterator
import glob
import os
import torch
import tempfile

class MuonDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str, 
                 batch_size: int = 128, 
                 num_workers: int = 0,
                 drop_empty_events: bool = False,
                 file_format: str = "hdf5",
                 federation_url: str = None,
                 token: str = None,
                 parquet_batch_reader: bool = False,
                 pin_memory: bool = False,
                 prefetch_factor: int = 2,
                 shuffle_parquet: bool = False,
                 multi_file_shuffle: int = 0,
                 files_override: list = None,
                 prefetch_ahead: int = 0,
                 prefetch_concurrency: int = 4,
                 prefetch_dir: str = None,
                 prefetch_batches: int = 0,
                 delete_after_use: bool = False,
                 limit_files_per_epoch: int = 0,
                 initial_processed_files: list = None,
                 token_refresh_args: dict = None
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["initial_processed_files"])
        self.files = []
        self.train_dataset = None
        self.prefetcher = None
        
        # Shared state for tracking processed files across workers
        import multiprocessing
        self._manager = multiprocessing.Manager()
        self.processed_files = self._manager.list(initial_processed_files if initial_processed_files else [])

    def state_dict(self):
        return {"processed_files": list(self.processed_files)}

    def load_state_dict(self, state_dict):
        if "processed_files" in state_dict:
            # Clear and refill the shared list
            while len(self.processed_files) > 0:
                self.processed_files.pop()
            self.processed_files.extend(state_dict["processed_files"])
            print(f"Restored {len(self.processed_files)} processed files from checkpoint.")
            self._cleanup_processed_files()

    def _cleanup_processed_files(self):
        """Delete local copies of files that are already marked as processed."""
        if not self.hparams.delete_after_use or not self.hparams.prefetch_dir:
            return
        
        from ..utils.pelican_utils import pelican_uri_to_local_cache_path
        cleaned_paths = []
        import os
        for uri in self.processed_files:
            # If it's a URI, convert to local path. If it's already a local path, use it.
            if uri.startswith("pelican://"):
                local_p = pelican_uri_to_local_cache_path(uri, cache_dir=self.hparams.prefetch_dir)
            else:
                local_p = uri
            
            if os.path.exists(local_p):
                try:
                    os.remove(local_p)
                    cleaned_paths.append(os.path.basename(local_p))
                except Exception:
                    pass
        
        if cleaned_paths:
            if len(cleaned_paths) > 10:
                summary = ", ".join(cleaned_paths[:5]) + " ... " + ", ".join(cleaned_paths[-5:])
                print(f"Cleaned up {len(cleaned_paths)} stale local files: {summary}")
            else:
                print(f"Cleaned up these already-processed files: {', '.join(cleaned_paths)}")

    def setup(self, stage=None):
        if self.train_dataset is not None:
             # If we are limiting files per epoch, we don't return early here
             # because we need to re-verify the file list for the next sub-epoch.
             if self.hparams.limit_files_per_epoch == 0:
                return

        # 1. Expand Files
        if self.hparams.files_override:
             self.files = self.hparams.files_override
        else:
             if "pelican://" in self.hparams.data_dir:
                 self.files, inferred_fed = expand_pelican_wildcards(
                     [self.hparams.data_dir], 
                     federation_url=self.hparams.federation_url, 
                     token=self.hparams.token
                 )
                 if not self.hparams.federation_url:
                     self.hparams.federation_url = inferred_fed
             elif os.path.isdir(self.hparams.data_dir):
                ext = "hdf5" if self.hparams.file_format == "hdf5" else "parquet"
                self.files = sorted(glob.glob(os.path.join(self.hparams.data_dir, f"**/*.{ext}"), recursive=True))
             else:
                self.files = sorted(glob.glob(self.hparams.data_dir))
        
        # 1b. Shuffle Files (Deterministic) if requested
        # We do this BEFORE filtering and BEFORE prefetcher init so that:
        # 1. The prefetcher sees the shuffled order and downloads "next" in that order.
        # 2. Resumption works because the seed makes the shuffle deterministic.
        if self.hparams.shuffle_parquet:
            import random
            # Use fixed seed for reproducibility across restarts
            random.Random(42).shuffle(self.files)

        # Filter out already processed files
        if self.processed_files:
            processed_set = set(self.processed_files)
            initial_count = len(self.files)
            
            # Keep track of what we are skipping
            self.files = [f for f in self.files if f not in processed_set]
            skipped = initial_count - len(self.files)
            if skipped > 0:
                print(f"Skipping {skipped}/{initial_count} already processed files.")
                self._cleanup_processed_files()

        if not self.files and not "pelican://" in self.hparams.data_dir: 
             print(f"Warning: No files found in {self.hparams.data_dir}")
             return

        # 2. Start Pelican Prefetcher if needed
        if self.hparams.prefetch_ahead > 0 and self.hparams.federation_url:
            p_dir = self.hparams.prefetch_dir or tempfile.gettempdir()
            # Convert to absolute path to avoid issues with DataLoader worker processes
            p_dir = os.path.abspath(p_dir)

            # Setup dynamic token refresh
            token_factory = None
            if self.hparams.token_refresh_args:
                from ..utils.pelican_utils import fetch_pelican_token_via_helper
                tr_args = self.hparams.token_refresh_args
                fed_url = self.hparams.federation_url
                
                # Define factory to call the helper (which uses the cache file)
                def _tf():
                    try:
                        return fetch_pelican_token_via_helper(
                            scope_path=tr_args.get("scope_path"),
                            federation_url=fed_url,
                            oidc_url=tr_args.get("oidc_url"),
                            auth_cache_file=tr_args.get("auth_cache_file"),
                            storage_prefix=tr_args.get("storage_prefix"),
                        )
                    except Exception as e:
                        print(f"Token refresh error in factory: {e}")
                        return None
                token_factory = _tf

            self.prefetcher = PelicanPrefetcher(
                self.files,
                federation_url=self.hparams.federation_url,
                token=self.hparams.token,
                cache_dir=p_dir,
                ahead=self.hparams.prefetch_ahead,
                concurrency=self.hparams.prefetch_concurrency,
                token_factory=token_factory
            )
            self.prefetcher.start()
            # Store original URIs and remap files to local cache paths
            self.original_uris = self.files.copy()
            self.files = [self.prefetcher.local_path(f) for f in self.files]
            
            # Wait for some files to be ready before starting training
            import time
            wait_count = min(self.hparams.multi_file_shuffle or 1, 10, len(self.files))
            print(f"Waiting for prefetcher to download first {wait_count} files (initially)...")
            start_t = time.time()
            max_wait = 600 # 10 minutes max wait
            while (time.time() - start_t) < max_wait:
                ready_count = sum(1 for f in self.files if os.path.exists(f))
                if ready_count >= wait_count:
                    break
                # Also break if the prefetcher has an error on one of the early files
                # (but that's harder to check here, we'll let it time out if stuck)
                time.sleep(1)
            
            ready_count = sum(1 for f in self.files if os.path.exists(f))
            print(f"Ready with {ready_count} files after {time.time() - start_t:.1f}s.")

        # 3. Instantiate Dataset
        if self.hparams.file_format == "hdf5":
             self.train_dataset = MultiHDF5Dataset(self.files)
        
        elif self.hparams.file_format in ["parquet", "hf"]:
             if self.hparams.parquet_batch_reader and self.hparams.file_format == "parquet":
                  # If prefetcher is active, files are local and don't need PelicanFS
                  use_pelican = (self.prefetcher is None)
                  # Pass original URIs if using prefetcher
                  original_uris = getattr(self, 'original_uris', None) if self.prefetcher else None
                  self.train_dataset = get_parquet_batch_dataset(
                        self.files if self.files else [self.hparams.data_dir],
                        batch_size=self.hparams.batch_size,
                        federation_url=self.hparams.federation_url if use_pelican else None,
                        token=self.hparams.token if use_pelican else None,
                        shuffle=self.hparams.shuffle_parquet,
                        multi_file_shuffle=self.hparams.multi_file_shuffle,
                        drop_empty_events=self.hparams.drop_empty_events,
                        prefetcher=self.prefetcher,
                        original_uris=original_uris,
                        delete_after_use=self.hparams.delete_after_use,
                        processed_files_shared=self.processed_files,
                        limit_files_per_epoch=self.hparams.limit_files_per_epoch
                  )
             else:
                 self.train_dataset = get_hf_dataset(
                     self.files if self.files else [self.hparams.data_dir],
                     file_format=self.hparams.file_format if self.hparams.file_format != "hf" else "parquet",
                     federation_url=self.hparams.federation_url,
                     token=self.hparams.token,
                     streaming=True
                 )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Dataset not initialized.")
        
        # We don't need a hard wait here anymore because ParquetBatchIterableDataset 
        # has its own wait-per-file logic.
            
        kwargs = {
            "batch_size": self.hparams.batch_size,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
        }
        
        if self.hparams.num_workers > 0:
             kwargs["prefetch_factor"] = self.hparams.prefetch_factor
             kwargs["persistent_workers"] = True

        # Check if using the vectorized parquet batch reader
        if self.hparams.parquet_batch_reader and self.hparams.file_format == "parquet":
             # Dataset already yields batches of tensors
             kwargs["batch_size"] = None
             loader = DataLoader(
                self.train_dataset,
                **kwargs
             )
        # HDF5
        elif self.hparams.file_format == "hdf5":
             loader = DataLoader(
                self.train_dataset,
                collate_fn=ragged_collate_fn,
                shuffle=True,
                **kwargs
             )
        else:
             # Streaming (HuggingFace or custom generator)
             loader = DataLoader(
                self.train_dataset,
                collate_fn=hf_collate_fn,
                **kwargs
             )
        
        # Wrap with background prefetcher iterator if requested
        if self.hparams.prefetch_batches > 0:
            return PrefetchIterator(loader, max_prefetch=self.hparams.prefetch_batches)
        
        return loader

    def teardown(self, stage=None):
        if self.prefetcher:
            self.prefetcher.stop()
