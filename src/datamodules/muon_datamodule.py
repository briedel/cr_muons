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
                 prefetch_dir: str = None,
                 prefetch_batches: int = 0
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.files = []
        self.train_dataset = None
        self.prefetcher = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
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

        if not self.files and not "pelican://" in self.hparams.data_dir: 
             print(f"Warning: No files found in {self.hparams.data_dir}")
             return

        # 2. Start Pelican Prefetcher if needed
        if self.hparams.prefetch_ahead > 0 and self.hparams.federation_url:
            p_dir = self.hparams.prefetch_dir or tempfile.gettempdir()
            # Convert to absolute path to avoid issues with DataLoader worker processes
            p_dir = os.path.abspath(p_dir)
            self.prefetcher = PelicanPrefetcher(
                self.files,
                federation_url=self.hparams.federation_url,
                token=self.hparams.token,
                cache_dir=p_dir,
                ahead=self.hparams.prefetch_ahead
            )
            self.prefetcher.start()
            # Store original URIs and remap files to local cache paths
            self.original_uris = self.files.copy()
            self.files = [self.prefetcher.local_path(f) for f in self.files]
            
            # IMPORTANT: Only use files that actually exist locally
            # The prefetcher downloads ahead, so we work with what's available
            # This prevents errors from trying to open files that haven't downloaded yet
            existing_files = [f for f in self.files if os.path.exists(f)]
            if not existing_files:
                # Wait a bit for initial files to download
                import time
                print(f"Waiting for prefetcher to download initial files...")
                for _ in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    existing_files = [f for f in self.files if os.path.exists(f)]
                    if existing_files:
                        break
            
            if not existing_files:
                raise RuntimeError(f"No files downloaded by prefetcher after 30s. Check prefetcher logs.")
            
            print(f"Using {len(existing_files)} files (out of {len(self.files)} total)")
            self.files = existing_files

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
                        prefetcher=self.prefetcher,
                        original_uris=original_uris
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
        
        # Wait for some files to be downloaded before starting
        if self.prefetcher:
            import time
            import os
            print(f"Waiting for prefetcher to download initial files...")
            wait_time = 0
            min_files = min(self.hparams.prefetch_ahead, 5)  # Wait for at least 5 files
            while wait_time < 300:  # Max 5 minutes
                # Check how many files are actually downloaded
                if hasattr(self, 'files') and self.files:
                    downloaded = sum(1 for f in self.files if os.path.exists(f))
                    if downloaded >= min_files:
                        print(f"✓ {downloaded} files downloaded, starting training")
                        break
                time.sleep(2)
                wait_time += 2
            else:
                print(f"Warning: Only {downloaded if 'downloaded' in locals() else 0} files available after {wait_time}s")
        
        # Force num_workers=0 when using prefetcher to avoid multiprocessing issues
        # (prefetcher runs in main process, workers can't access downloaded files)
        num_workers = 0 if self.prefetcher else self.hparams.num_workers
            
        kwargs = {
            "batch_size": self.hparams.batch_size,
            "num_workers": num_workers,
            "pin_memory": self.hparams.pin_memory,
        }
        
        if num_workers > 0:
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
