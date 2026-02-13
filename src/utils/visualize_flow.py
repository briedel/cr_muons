import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import pandas as pd
from tqdm import tqdm
from src.models.flow_module import MuonFlow
from src.datamodules.muon_datamodule import MuonDataModule

# Configure Matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MuonFlow results")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test .parquet files (glob pattern)")
    parser.add_argument("--output_dir", type=str, default="plots_flow", help="Where to save plots")
    parser.add_argument("--num_events", type=int, default=1000, help="Number of events to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()

def plot_histograms(real, gen, output_dir, prefix=""):
    """
    Plot overlay histograms for Real vs Generated features.
    Assumes features are: [Energy, X, Y] for feat_dim=3
    or [Energy, X, Y, Z, Time] for feat_dim=5 (adjust as needed)
    """
    # Determine feature dimensionality and names based on shape
    feat_dim = real.shape[1]
    
    # Common feature names (adjust if your model uses specific ones like Energy, R, Z)
    # The config usually is 'xy' -> [E, x, y] or 'r' -> [E, r]
    if feat_dim == 2:
         # Likely [Energy, Radius] or similar if r-mode
         labels = ["Log10(Energy)", "Radius / Other"] 
    elif feat_dim == 3:
        labels = ["Log10(Energy) [GeV]", "X [m]", "Y [m]"]
    else:
        labels = [f"Feature {i}" for i in range(feat_dim)]

    for i in range(feat_dim):
        plt.figure()
        
        # Get data range for shared binning
        r_data = real[:, i]
        g_data = gen[:, i]
        
        # Handle log energy for better plotting if it looks exponential
        # (Assuming the model outputs physical values, E is usually widely distributed)
        # But here we just plot what we get.
        
        # Determine strict bounds for histograms to avoid outliers skewing the view
        # Use percentiles
        low = np.percentile(np.concatenate([r_data, g_data]), 1)
        high = np.percentile(np.concatenate([r_data, g_data]), 99)
        bins = np.linspace(low, high, 50)
        
        plt.hist(r_data, bins=bins, alpha=0.5, label='Real', density=True, color='blue', edgecolor='k')
        plt.hist(g_data, bins=bins, alpha=0.5, label='Generated', density=True, color='orange', edgecolor='k')
        
        plt.xlabel(labels[i])
        plt.ylabel("Density")
        plt.title(f"Distribution: {labels[i]} ({prefix})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        fname = f"{output_dir}/hist_{prefix}_feat_{i}.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")

def plot_multiplicity(real_counts, gen_counts, output_dir):
    plt.figure()
    
    max_val = max(np.max(real_counts), np.max(gen_counts))
    if max_val > 500: # If huge counts, log scale or larger bins
        bins = np.logspace(0, np.log10(max_val), 50)
        plt.xscale('log')
    else:
        bins = np.arange(0, max_val + 2)
        
    plt.hist(real_counts, bins=bins, alpha=0.5, label='Real', density=True, color='blue')
    plt.hist(gen_counts, bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
    
    plt.xlabel("Number of Muons")
    plt.ylabel("Probability")
    plt.title("Multiplicity Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fname = f"{output_dir}/hist_multiplicity.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

def plot_scatter(real, gen, output_dir):
    """
    Plot 2D correlations (e.g. Energy vs Radius)
    """
    if real.shape[1] < 2:
        return

    # Plot Feat 0 vs Feat 1 (e.g. Energy vs Radius/X)
    plt.figure(figsize=(16, 6))
    
    # Real
    plt.subplot(1, 2, 1)
    plt.scatter(real[:, 1], real[:, 0], s=1, alpha=0.1, c='blue')
    plt.xlabel("Feature 1 (Spatial?)")
    plt.ylabel("Feature 0 (Energy?)")
    plt.title("Real Correlations")
    plt.grid(True, alpha=0.3)
    
    # Gen
    plt.subplot(1, 2, 2)
    plt.scatter(gen[:, 1], gen[:, 0], s=1, alpha=0.1, c='orange')
    plt.xlabel("Feature 1 (Spatial?)")
    plt.ylabel("Feature 0 (Energy?)")
    plt.title("Generated Correlations")
    plt.grid(True, alpha=0.3)
    
    fname = f"{output_dir}/scatter_feat0_vs_feat1.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Model
    print(f"Loading checkpoint: {args.ckpt_path}")
    try:
        model = MuonFlow.load_from_checkpoint(args.ckpt_path)
        model.to(args.device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # 2. Setup Data
    # Use the DataModule to ensure consistent preprocessing
    print("Setting up DataModule...")
    # Find files locally
    files = sorted(glob.glob(args.data_dir))
    if not files:
        print(f"No files found matching {args.data_dir}")
        return
        
    # We create a dummy DM just to get the dataloader logic
    dm = MuonDataModule(
        data_dir=args.data_dir, # This might be ignored if we pass files_override
        batch_size=args.batch_size,
        files_override=files[:5], # Just take a few files
        file_format="parquet",
        parquet_batch_reader=True,
        muon_feature_selection=model.hparams.muon_feature_selection # Match model config
    )
    dm.setup()
    loader = dm.train_dataloader() # Use train loader as it's the same format
    
    # 3. Generation Loop
    all_real_muons = []
    all_gen_muons = []
    all_real_counts = []
    all_gen_counts = []
    
    events_processed = 0
    print(f"Generating samples for ~{args.num_events} events...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # Batch: flat_muons, batch_idx, prims, counts
            # Note: flat_muons are raw (physical units) here because normalize happens inside model
            # BUT wait, the dataloader might output standardized data if configured? 
            # Reviewing code: DM yields raw. Model.training_step handles normalization.
            
            real_muons_raw, batch_idx, prims, real_counts = batch
            
            real_muons_raw = real_muons_raw.to(args.device)
            prims = prims.to(args.device)
            
            # Predict
            # predict_step returns LIST of tensors (one per event) typically
            # The model's predict_step returns DENORMALIZED (PHYSICAL) samples.
            gen_samples_list = model.predict_step(prims, batch_idx=None)
            
            # Collect Real Data
            # We need to split the flattened real muons back into events to match structure if we want
            # strictly event-by-event comparison, or just concat all for histograms.
            # However, for plotting histograms, we just need the bulk distribution.
            
            # Filter real muons to match feature selection?
            # Model handles normalization inside, but `real_muons_raw` from loader is
            # [N, 5] (E, x, y, z, t) usually.
            # If model uses 'xy', it slices inside. We should slice here too for comparison.
            
            feat_dim = model.hparams.feat_dim
            if real_muons_raw.shape[1] == 5 and feat_dim == 3:
                 # Standard conversion: usually indices 0=E, 1=t?, 2=x, 3=y, 4=z?
                 # Wait, let's check input format.
                 # inspect_parquet usually shows columns.
                 # Assuming typical: E, Z, X, Y, ...
                 # Actually, let's rely on what the model does training_step:
                 # if flat_muons.shape[1] == 5 and self.hparams.feat_dim == 3: muons_raw = flat_muons[:, 2:]
                 real_muons_selected = real_muons_raw[:, 2:]
            elif real_muons_raw.shape[1] > feat_dim:
                 # Fallback: take last N
                 real_muons_selected = real_muons_raw[:, -feat_dim:]
            else:
                 real_muons_selected = real_muons_raw

            all_real_muons.append(real_muons_selected.cpu().numpy())
            all_real_counts.append(real_counts.cpu().numpy())
            
            # Collect Gen Data
            # gen_samples_list is list of tensors. Concat them.
            batch_gen_muons = torch.cat(gen_samples_list, dim=0)
            all_gen_muons.append(batch_gen_muons.cpu().numpy())
            
            # Count per event
            batch_gen_counts = [t.shape[0] for t in gen_samples_list]
            all_gen_counts.extend(batch_gen_counts)
            
            events_processed += prims.shape[0]
            if events_processed >= args.num_events:
                break
                
    # 4. Concatenate
    real_muons_np = np.concatenate(all_real_muons, axis=0)
    gen_muons_np = np.concatenate(all_gen_muons, axis=0)
    real_counts_np = np.concatenate(all_real_counts, axis=0)
    gen_counts_np = np.array(all_gen_counts)
    
    print(f"Processed: {events_processed} Events")
    print(f"Real Muons: {real_muons_np.shape[0]}")
    print(f"Gen Muons:  {gen_muons_np.shape[0]}")
    
    # 5. Plot
    print("Plotting histograms...")
    plot_histograms(real_muons_np, gen_muons_np, args.output_dir)
    plot_multiplicity(real_counts_np, gen_counts_np, args.output_dir)
    plot_scatter(real_muons_np, gen_muons_np, args.output_dir)
    
    print(f"Done! Results in {args.output_dir}")

if __name__ == "__main__":
    main()
