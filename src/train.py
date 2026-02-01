import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodules.muon_datamodule import MuonDataModule
from src.models.gan_module import MuonGAN
from src.models.flow_module import MuonFlow
from src.callbacks.adaptive_tuning import AdaptiveCriticTuning
from src.callbacks.monitoring import PerformanceMonitoringCallback, HistogramLoggingCallback
from src.utils import expand_pelican_wildcards, fetch_pelican_token_via_helper
import argparse
import os
import time
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to HDF5 files (supports glob) or Pelican URL")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="gan", choices=["gan", "flow"])
    parser.add_argument("--drop_empty", action="store_true")
    
    # Dataset Params
    parser.add_argument("--file_format", type=str, default="parquet", choices=["hdf5", "parquet", "hf"], help="Data file format")
    parser.add_argument("--no_parquet_batch_reader", action="store_false", dest="parquet_batch_reader", help="Disable efficient Parquet batch reader")
    parser.set_defaults(parquet_batch_reader=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--shuffle_parquet", action="store_true")
    parser.add_argument("--multi_file_shuffle", type=int, default=0, help="Number of files to interleave batches from")
    parser.add_argument("--prefetch_batches", type=int, default=0, help="Number of batches to prefetch in background")
    parser.add_argument("--prefetch_ahead", type=int, default=0, help="Number of Pelican files to prefetch ahead")
    parser.add_argument("--prefetch_dir", type=str, default=None, help="Cache directory for Pelican prefetching")

    # Pelican / Authtokens
    parser.add_argument("--auto_token", action="store_true", help="Automatically fetch via osg-token-scope")
    parser.add_argument("--federation_url", type=str, default=None)
    parser.add_argument("--pelican_scope_path", type=str, default=None)
    parser.add_argument("--pelican_storage_prefix", type=str, default="/icecube")
    parser.add_argument("--pelican_oidc_url", type=str, default=None)
    parser.add_argument("--pelican_auth_cache_file", type=str, default="pelican_auth_cache.json")


    # GAN / Flow shared params
    parser.add_argument("--cond_dim", type=int, default=5, help="5 features: [E, zenith, mass, time, depth]")
    parser.add_argument("--feat_dim", type=int, default=4, help="4 features: [E, x, y, time]")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--critic_pooling", type=str, default="amax")

    # Flow specific params
    parser.add_argument("--flow_bins", type=int, default=10)
    parser.add_argument("--flow_transforms", type=int, default=3)
    parser.add_argument("--mult_loss_weight", type=float, default=0.1)

    parser.add_argument("--latent_dim_global", type=int, default=32)
    parser.add_argument("--latent_dim_local", type=int, default=16)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--critic_steps", type=int, default=5)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--gp_every", type=int, default=2)
    parser.add_argument("--gp_max_pairs", type=int, default=4096)
    parser.add_argument("--gp_sample_fraction", type=float, default=0.0)
    parser.add_argument("--max_muons_per_batch", type=int, default=0)
    parser.add_argument("--max_muons_per_event", type=int, default=0)
    parser.add_argument("--outliers_dir", type=str, default=None)
    
    # TensorBoard / Logging
    parser.add_argument("--tb_logdir", type=str, default="logs_tensorboard", help="Directory for TensorBoard logs")
    parser.add_argument("--tb_run_name", type=str, default=None, help="Optional run name for TensorBoard logs")
    parser.add_argument("--tb_log_interval", type=int, default=10)
    parser.add_argument("--tb_hist_interval", type=int, default=1000)
    parser.add_argument("--tb_max_muons", type=int, default=20000)
    
    # Checkpointing
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save checkpoints (deprecated, use pl default)")
    parser.add_argument("--model_checkpoint", type=str, default=None)

    args = parser.parse_args()

    pl.seed_everything(42)

    # Token Logic
    token = None
    if args.auto_token or args.federation_url:
        print("Handling Pelican Token...")
        # Simple logic: if auto_token, try to fetch
        if args.auto_token:
            from urllib.parse import urlparse
            
            scope = args.pelican_scope_path
            # Infer scope from data_dir if it's a pelican URL
            if not scope and "pelican://" in args.data_dir:
                 # pelican://host/path/to/data -> /path/to/data
                 parsed = urlparse(args.data_dir.replace("pelican://", "http://"))
                 # This is a hacky parse, assuming utility handles prefixes better
                 # Using the fallback logic from training/train.py logic would be better if imported
                 # But let's assume util handles it or default prefix logic works
                 pass
            
            # Using the util function
            token = fetch_pelican_token_via_helper(
                scope_path=getattr(args, "pelican_scope_path", None), # Util handles inference if None? Need to check.
                federation_url=args.federation_url,
                oidc_url=args.pelican_oidc_url,
                auth_cache_file=args.pelican_auth_cache_file,
                storage_prefix=args.pelican_storage_prefix
            )
            print(f"Token fetched: {'Yes' if token else 'No'}")

    # File Expansion
    files = None
    if "pelican://" in args.data_dir:
         print(f"Expanding wildcards for: {args.data_dir}")
         expanded_files, inferred_fed = expand_pelican_wildcards([args.data_dir], federation_url=args.federation_url, token=token)
         files = expanded_files
         if not args.federation_url and inferred_fed:
             args.federation_url = inferred_fed
         print(f"Found {len(files)} files via Pelican.")
    

    # DataModule
    dm = MuonDataModule(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        drop_empty_events=args.drop_empty,
        file_format=args.file_format,
        parquet_batch_reader=args.parquet_batch_reader,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        shuffle_parquet=args.shuffle_parquet,
        multi_file_shuffle=args.multi_file_shuffle,
        federation_url=args.federation_url,
        token=token,
        files_override=files,
        prefetch_ahead=args.prefetch_ahead,
        prefetch_dir=args.prefetch_dir,
        prefetch_batches=args.prefetch_batches
    )

    # Model
    if args.model_type == "gan":
        model = MuonGAN(
            cond_dim=args.cond_dim,
            feat_dim=args.feat_dim,
            hidden_dim=args.hidden_dim,
            critic_pooling=args.critic_pooling,
            latent_dim_global=args.latent_dim_global,
            latent_dim_local=args.latent_dim_local,
            lr=args.lr,
            lambda_gp=args.lambda_gp,
            critic_steps=args.critic_steps,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip_norm=args.grad_clip_norm,
            gp_every=args.gp_every,
            gp_max_pairs=args.gp_max_pairs,
            gp_sample_fraction=args.gp_sample_fraction,
            max_muons_per_batch=args.max_muons_per_batch,
            max_muons_per_event=args.max_muons_per_event,
            outliers_dir=args.outliers_dir
        )
        callbacks = [
            ModelCheckpoint(filename="{epoch}-{g_loss:.2f}", monitor="g_loss", mode="min", save_last=True),
            LearningRateMonitor(logging_interval="step"),
            AdaptiveCriticTuning(),
            PerformanceMonitoringCallback(log_interval=args.tb_log_interval),
            HistogramLoggingCallback(log_every_n_steps=args.tb_hist_interval, max_muons=args.tb_max_muons)
        ]
    elif args.model_type == "flow":
        model = MuonFlow(
            cond_dim=args.cond_dim,
            feat_dim=args.feat_dim,
            hidden_dim=args.hidden_dim,
            bins=args.flow_bins,
            transforms=args.flow_transforms,
            mult_loss_weight=args.mult_loss_weight,
            lr=args.lr
        )
        callbacks = [
            ModelCheckpoint(monitor="val_loss", mode="min", save_last=True),
            LearningRateMonitor(logging_interval="step")
        ]

    # Trainer
    run_name = args.tb_run_name or time.strftime("%Y%m%d-%H%M%S")
    logger = TensorBoardLogger(save_dir=args.tb_logdir, name=run_name, version="")
    
    # Log args to TB
    logger.log_hyperparams(vars(args))
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
