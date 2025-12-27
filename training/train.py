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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    fs = get_filesystem(args.federation_url, args.token)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", 
                        "--infiles", 
                        nargs='+', 
                        type=Path, 
                        required=True)
    parser.add_argument("--use-hf", 
                        action="store_true", 
                        help="Use Hugging Face Streaming Dataset")
    parser.add_argument("--federation-url", 
                        type=str, 
                        default=None,
                        help="Pelican Federation URL (e.g. pelican://osg-htc.org)")
    parser.add_argument("--token", 
                        type=str, 
                        default=None,
                        help="Auth token for Pelican")
    parser.add_argument("--checkpoint", 
                        type=str, 
                        default="training_checkpoint.json",
                        help="Path to checkpoint file tracking processed files")
    parser.add_argument("--model-checkpoint", 
                        type=str, 
                        default="model_checkpoint.pt",
                        help="Path to model checkpoint file")
    
    # Model Hyperparameters
    parser.add_argument("--cond-dim", 
                        type=int, 
                        default=4, 
                        help="Dimension of event conditions")
    parser.add_argument("--feat-dim", 
                        type=int, 
                        default=3, 
                        help="Dimension of muon features")
    parser.add_argument("--latent-dim-global", 
                        type=int, 
                        default=32, 
                        help="Global latent dimension")
    parser.add_argument("--latent-dim-local", 
                        type=int, 
                        default=16, 
                        help="Local latent dimension")
    parser.add_argument("--hidden-dim", 
                        type=int, 
                        default=256, 
                        help="Hidden dimension size")
    parser.add_argument("--lambda-gp", 
                        type=float, 
                        default=10.0, 
                        help="Gradient penalty weight")

    args=parser.parse_args()

    main(args)

