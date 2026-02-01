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
            self.prefetcher = PelicanPrefetcher(
                self.files,
                federation_url=self.hparams.federation_url,
                token=self.hparams.token,
                cache_dir=p_dir,
                ahead=self.hparams.prefetch_ahead
            )
            self.prefetcher.start()
            # Remap files to local cache paths
            self.files = [self.prefetcher.local_path(f) for f in self.files]

        # 3. Instantiate Dataset
        if self.hparams.file_format == "hdf5":
             self.train_dataset = MultiHDF5Dataset(self.files)
        
        elif self.hparams.file_format in ["parquet", "hf"]:
             if self.hparams.parquet_batch_reader and self.hparams.file_format == "parquet":
                  self.train_dataset = get_parquet_batch_dataset(
                        self.files if self.files else [self.hparams.data_dir],
                        batch_size=self.hparams.batch_size,
                        federation_url=self.hparams.federation_url,
                        token=self.hparams.token,
                        shuffle=self.hparams.shuffle_parquet,
                        multi_file_shuffle=self.hparams.multi_file_shuffle
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
