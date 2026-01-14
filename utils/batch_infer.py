#!/usr/bin/env python3
import argparse
import os
import sys
import math
import time
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from training.model import ScalableGenerator
from training.normalizer import DataNormalizer


def load_generator(checkpoint_path: str, device: str = "cpu") -> ScalableGenerator:
    gen = ScalableGenerator(device=device)
    gen.to(device)
    gen.eval()
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict):
                if "gen_state_dict" in ckpt:
                    gen.load_state_dict(ckpt["gen_state_dict"])
                elif "generator" in ckpt and isinstance(ckpt["generator"], dict):
                    gen.load_state_dict(ckpt["generator"])
                elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                    gen.load_state_dict(ckpt["state_dict"])
                else:
                    # Attempt direct state_dict load
                    gen.load_state_dict(ckpt)
            else:
                # Unknown format; skip load
                pass
            print(f"Loaded generator weights from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint '{checkpoint_path}': {e}")
    else:
        print("No checkpoint provided or file not found; using randomly initialized generator.")
    return gen


def read_primaries_parquet(path: str, batch_size: int = 0):
    """Read primaries from a Parquet file.

    Expects at least 4 columns: [E_GeV, Zenith_Rad, Mass_A, Depth_m].
    If additional columns exist, they will be ignored beyond the first 5 (to allow optional time).
    Returns a torch.Tensor [N, F].
    """
    table = pq.read_table(path)
    # Flatten to numpy; support variable number of columns
    cols = [table.column(i).to_pylist() for i in range(table.num_columns)]
    # Transpose rows
    rows = list(zip(*cols))
    import numpy as np
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Parquet must have >=4 columns [E, zenith, mass, depth]; got shape={arr.shape}")
    return torch.from_numpy(arr)


def write_muons_parquet(out_path: str, muons: torch.Tensor, batch_index: torch.Tensor, denormalize: bool):
    """Write generated muons to Parquet.
    Columns: event_index, E_GeV, X_m, Y_m
    """
    if muons is None or muons.numel() == 0:
        # Write empty table
        schema = pa.schema([
            pa.field("event_index", pa.int64()),
            pa.field("E_GeV", pa.float32()),
            pa.field("X_m", pa.float32()),
            pa.field("Y_m", pa.float32()),
        ])
        empty = pa.table({"event_index": [], "E_GeV": [], "X_m": [], "Y_m": []}, schema=schema)
        pq.write_table(empty, out_path)
        return

    mu = muons.detach().cpu()
    idx = batch_index.detach().cpu().long()

    if denormalize:
        # Convert normalized features back to physical units
        norm = DataNormalizer()
        mu_phys = norm.denormalize_features(mu)
        E = mu_phys[:, 0].numpy().astype("float32")
        X = mu_phys[:, 1].numpy().astype("float32")
        Y = mu_phys[:, 2].numpy().astype("float32")
    else:
        E = mu[:, 0].numpy().astype("float32")
        X = mu[:, 1].numpy().astype("float32")
        Y = mu[:, 2].numpy().astype("float32")

    event_idx = idx.numpy().astype("int64")
    table = pa.table({
        "event_index": event_idx,
        "E_GeV": E,
        "X_m": X,
        "Y_m": Y,
    })
    pq.write_table(table, out_path)


def predict_counts(gen: ScalableGenerator, cond_norm: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mul_log10 = gen.multiplicity_net(cond_norm)
        mul_log10 = torch.nan_to_num(mul_log10, nan=0.0, posinf=0.0, neginf=0.0)
        counts = torch.pow(10.0, mul_log10.squeeze(1)) - 1.0
        counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        counts = counts.clamp_min(0).round().long()
        return counts


def main():
    ap = argparse.ArgumentParser(description="Batch inference: generate muons from primaries")
    ap.add_argument("--input", required=True, help="Input Parquet with primaries")
    ap.add_argument("--output", required=True, help="Output Parquet for generated muons")
    ap.add_argument("--model-checkpoint", default="./model_checkpoint.pt", help="Path to trained generator checkpoint")
    ap.add_argument("--device", default="cuda", help="Device to run inference (cuda/cpu)")
    ap.add_argument("--counts-mode", default="predict", choices=["predict", "column", "constant"], help="How to choose per-event counts")
    ap.add_argument("--counts-column", default=None, help="Column name in input Parquet when counts-mode=column")
    ap.add_argument("--count-constant", type=int, default=0, help="Constant count when counts-mode=constant")
    ap.add_argument("--max-muons-per-event", type=int, default=0, help="Optional cap per event; 0 disables")
    ap.add_argument("--denormalize", action="store_true", help="Write outputs in physical units (E_GeV, X_m, Y_m)")

    args = ap.parse_args()

    device = args.device
    gen = load_generator(args.model_checkpoint, device=device)

    # Read primaries and normalize
    prims = read_primaries_parquet(args.input)
    prims = prims.to(device)
    norm = DataNormalizer()
    cond_norm = norm.normalize_primaries(prims)

    # Determine counts
    if args.counts_mode == "predict":
        counts = predict_counts(gen, cond_norm)
    elif args.counts_mode == "constant":
        counts = torch.full((cond_norm.size(0),), int(args.count_constant), dtype=torch.long, device=device)
    elif args.counts_mode == "column":
        if not args.counts_column:
            print("Error: --counts-column must be provided when --counts-mode=column", file=sys.stderr)
            sys.exit(2)
        table = pq.read_table(args.input, columns=[args.counts_column])
        arr = table.column(0).to_numpy().astype("int64")
        counts = torch.from_numpy(arr).to(device)
    else:
        print(f"Unknown counts-mode: {args.counts_mode}", file=sys.stderr)
        sys.exit(2)

    # Apply cap
    if int(args.max_muons_per_event) > 0:
        counts = torch.minimum(counts, torch.tensor(int(args.max_muons_per_event), device=device))
    counts = torch.clamp(counts, min=0)

    # Generate muons
    with torch.no_grad():
        flat_muons, batch_index = gen.generate_with_counts(cond_norm, counts)

    # Write outputs
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    write_muons_parquet(args.output, flat_muons, batch_index, denormalize=bool(args.denormalize))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
