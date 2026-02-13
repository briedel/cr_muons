import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodules.muon_datamodule import MuonDataModule
from src.models.gan_module import MuonGAN
from src.models.flow_module import MuonFlow
from src.callbacks.adaptive_tuning import AdaptiveCriticTuning
from src.callbacks.monitoring import PerformanceMonitoringCallback, HistogramLoggingCallback, PhysicalCorrectnessCallback, LearningRateLogger
from src.utils import (
    expand_pelican_wildcards,
    fetch_pelican_token_via_helper,
    infer_scope_path_from_pelican_uri,
    is_pelican_path,
)
import argparse
import time
import torch 
import os
import glob

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
    parser.add_argument("--prefetch_concurrency", type=int, default=4, help="Number of concurrent Pelican downloads")
    parser.add_argument("--prefetch_dir", type=str, default=None, help="Cache directory for Pelican prefetching")
    parser.add_argument("--delete_after_use", action="store_true", help="Delete local files after they have been processed")
    parser.add_argument("--limit_files_per_epoch", type=int, default=0, help="If > 0, stop each epoch after this many files")
    parser.add_argument("--muon_feature_selection", type=str, default="all", choices=["all", "xy", "r"], help="Select muon features: 'all', 'xy' (E,x,y), or 'r' (E,r)")

    # Pelican / Authtokens
    parser.add_argument("--auto_token", action="store_true", help="Automatically fetch via osg-token-scope")
    parser.add_argument("--federation_url", type=str, default=None)
    parser.add_argument("--pelican_scope_path", type=str, default=None)
    parser.add_argument("--pelican_storage_prefix", type=str, default="/icecube/wipac")
    parser.add_argument("--pelican_oidc_url", type=str, default="https://token-issuer.icecube.aq")
    parser.add_argument("--pelican_auth_cache_file", type=str, default="pelican_auth_cache.json")


    # GAN / Flow shared params
    parser.add_argument("--cond_dim", type=int, default=4, help="4 features: [E, zenith, mass, depth]")
    parser.add_argument("--feat_dim", type=int, default=3, help="3 features: [E, x, y]")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--critic_pooling", type=str, default="amax")

    # Flow specific params
    parser.add_argument("--flow_bins", type=int, default=10)
    parser.add_argument("--flow_transforms", type=int, default=3)
    parser.add_argument("--mult_loss_weight", type=float, default=0.1)
    parser.add_argument("--base_dist", type=str, default="normal", choices=["normal", "student-t"], help="Base distribution for flow")
    parser.add_argument("--student_dof", type=float, default=4.0, help="Degrees of freedom for Student-T base distribution")

    # GAN specific params
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
    parser.add_argument("--preflight_muon_threshold", type=int, default=0, help="Alias/override for max_muons_per_batch")
    parser.add_argument("--max_muons_per_event", type=int, default=0)
    parser.add_argument("--outliers_dir", type=str, default=None)
    
    # TensorBoard / Logging
    parser.add_argument("--tb_logdir", type=str, default="logs_tensorboard", help="Directory for TensorBoard logs")
    parser.add_argument("--tb_run_name", type=str, default=None, help="Optional run name for TensorBoard logs")
    parser.add_argument("--tb_log_interval", type=int, default=10)
    parser.add_argument("--tb_hist_interval", type=int, default=1000)
    parser.add_argument("--tb_max_muons", type=int, default=20000)
    parser.add_argument("--physics_check_interval", type=int, default=1000, help="Check physical correctness every N steps")
    
    # Checkpointing / Resume
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file (.ckpt) to resume from")
    parser.add_argument("--resume_last", action="store_true", help="Automatically resume from the latest checkpoint in tb_logdir")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=1000, help="Save checkpoint every N training steps")
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug prints for normalization")

    # Optimization
    parser.add_argument("--precision", type=str, default="32", 
                        help="Trainer precision. Common options: '32', '16-mixed' (FP16 mixed), 'bf16-mixed' (BF16 mixed), "
                             "'16-true', 'bf16-true', '64-true'. See PL docs for full list.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients over k batches before stepping optimizer")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for graph optimization")
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="Mode for torch.compile")

    args = parser.parse_args()

    # Handle preflight threshold alias
    if args.preflight_muon_threshold > 0:
        args.max_muons_per_batch = args.preflight_muon_threshold

    pl.seed_everything(42)

    # Token Logic
    token = None
    token_scope_path = None
    if args.auto_token or args.federation_url:
        print("Handling Pelican Token...")
        # Simple logic: if auto_token, try to fetch
        if args.auto_token:
            token_scope_path = args.pelican_scope_path
            if not token_scope_path and is_pelican_path(args.data_dir):
                token_scope_path = infer_scope_path_from_pelican_uri(
                    args.data_dir,
                    storage_prefix=args.pelican_storage_prefix
                )
            
            if not token_scope_path:
                print("Warning: --auto-token set but no scope path provided or inferred.")
            
            print(f"Fetching Pelican token for scope: {token_scope_path}")
            # Using the util function
            token = fetch_pelican_token_via_helper(
                scope_path=token_scope_path,
                federation_url=args.federation_url,
                oidc_url=args.pelican_oidc_url,
                auth_cache_file=args.pelican_auth_cache_file,
                storage_prefix=args.pelican_storage_prefix
            )
            print(f"Token fetched: {'Yes' if token else 'No'}")

    token_refresh_args = None
    if args.auto_token and token_scope_path:
        token_refresh_args = {
            "scope_path": token_scope_path,
            "oidc_url": args.pelican_oidc_url,
            "auth_cache_file": args.pelican_auth_cache_file,
            "storage_prefix": args.pelican_storage_prefix
        }

    # File Expansion
    files = None
    if "pelican://" in args.data_dir:
         print(f"Expanding wildcards for: {args.data_dir}")
         expanded_files, inferred_fed = expand_pelican_wildcards([args.data_dir], federation_url=args.federation_url, token=token)
         files = expanded_files
         if not args.federation_url and inferred_fed:
             args.federation_url = inferred_fed
         print(f"Found {len(files)} files via Pelican.")
    


    # Resume Logic - Pre-calculate to extract DataModule state
    ckpt_path = args.checkpoint
    initial_processed_files = None
    if args.resume_last:
        # Look for checkpoints in the log directory
        ckpt_search = os.path.join(args.tb_logdir, "**", "checkpoints", "last.ckpt")
        ckpts = glob.glob(ckpt_search, recursive=True)
        if ckpts:
            # Get the most recently modified one
            ckpt_path = max(ckpts, key=os.path.getmtime)
            print(f"Auto-resuming from: {ckpt_path}")

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            print(f"Peeking into checkpoint {ckpt_path} to restore processed files list...")
            # Map location cpu to avoid OOM or CUDA init issues just for this
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            if "datamodule_state_dict" in checkpoint:
                dm_state = checkpoint["datamodule_state_dict"]
                if "processed_files" in dm_state:
                    initial_processed_files = dm_state["processed_files"]
            elif "MuonDataModule" in checkpoint:
                # Fallback: sometimes Lightning saves it under the class name
                dm_state = checkpoint["MuonDataModule"]
                if "processed_files" in dm_state:
                    initial_processed_files = dm_state["processed_files"]
            
            if initial_processed_files:
                print(f"  -> Found {len(initial_processed_files)} previously processed files to skip.")
        except Exception as e:
            print(f"Warning: Failed to peek at checkpoint: {e}")

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
        prefetch_concurrency=args.prefetch_concurrency,
        prefetch_dir=args.prefetch_dir,
        prefetch_batches=args.prefetch_batches,
        delete_after_use=args.delete_after_use,
        limit_files_per_epoch=args.limit_files_per_epoch,
        initial_processed_files=initial_processed_files,
        muon_feature_selection=args.muon_feature_selection,
        token_refresh_args=token_refresh_args
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
            drop_empty_events=args.drop_empty,
            outliers_dir=args.outliers_dir
        )
        callbacks = [
            ModelCheckpoint(filename="{epoch}-{g_loss:.2f}", monitor="g_loss", mode="min", save_last=True, every_n_train_steps=args.checkpoint_every_n_steps),
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
            lr=args.lr,
            chunk_size=args.max_muons_per_batch,
            debug=args.debug,
            base_dist=args.base_dist,
            student_dof=args.student_dof
        )
        callbacks = [
            ModelCheckpoint(filename="{epoch}-{step}-{train_loss:.2f}", monitor="train_loss", mode="min", save_last=True, every_n_train_steps=args.checkpoint_every_n_steps),
            LearningRateMonitor(logging_interval="step"),
            PerformanceMonitoringCallback(log_interval=args.tb_log_interval),
            PhysicalCorrectnessCallback(log_every_n_steps=args.physics_check_interval),
            LearningRateLogger()
        ]

    # Optional: Torch Compile
    if args.compile:
        if not hasattr(torch, "compile"):
             print("Warning: --compile requested but torch.compile is not available (requires PyTorch 2.0+). Ignoring.")
        else:
             print(f"Compiling model submodules with mode='{args.compile_mode}'...")
             # Instead of compiling the LightningModule (which can be problematic with Trainer checks),
             # we compile the underlying neural networks.
             if args.model_type == "flow":
                 # Compile the flow and multiplicity nets
                 if hasattr(model, "flow"):
                     model.flow = torch.compile(model.flow, mode=args.compile_mode)
                 if hasattr(model, "multiplicity_net"):
                     model.multiplicity_net = torch.compile(model.multiplicity_net, mode=args.compile_mode)
             elif args.model_type == "gan":
                 # Compile generator and critic
                 if hasattr(model, "generator"):
                     model.generator = torch.compile(model.generator, mode=args.compile_mode)
                 if hasattr(model, "critic"):
                     model.critic = torch.compile(model.critic, mode=args.compile_mode)

    # Precision Checks
    if "bf16" in args.precision and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print(f"Warning: Precision '{args.precision}' requested, but BF16 is not supported by this GPU. "
              "Training may fail or fallback to FP32 depending on PyTorch Lightning settings.")

    # Trainer
    run_name = args.tb_run_name or time.strftime("%Y%m%d-%H%M%S")
    logger = TensorBoardLogger(save_dir=args.tb_logdir, name=run_name, version="")
    
    # Log args to TB
    logger.log_hyperparams(vars(args))
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
