import h5py
import numpy as np
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def convert_h5_to_parquet(h5_path, parquet_path, batch_size=10000):
    print(f"Converting {h5_path} to {parquet_path}...")
    
    # Define Schema
    # primary: list of floats
    # muons: list of list of floats (nested list)
    schema = pa.schema([
        ('primary', pa.list_(pa.float32())),
        ('muons', pa.list_(pa.list_(pa.float32())))
    ])
    
    with h5py.File(h5_path, 'r') as f:
        primaries_ds = f['primaries']
        muons_ds = f['muons']
        counts_ds = f['counts']
        
        total_events = counts_ds.shape[0]
        writer = None
        muon_offset = 0
        
        for i in range(0, total_events, batch_size):
            # Read batch of data into memory
            # Slicing h5py datasets reads into memory as numpy arrays
            batch_counts = counts_ds[i : i + batch_size]
            batch_primaries = primaries_ds[i : i + batch_size]
            
            n_batch = len(batch_counts)
            total_muons_in_batch = np.sum(batch_counts)
            
            # Read corresponding muons
            batch_muons = muons_ds[muon_offset : muon_offset + total_muons_in_batch]
            muon_offset += total_muons_in_batch
            
            # Prepare data for PyArrow
            primary_list = []
            muons_list = []
            
            local_muon_idx = 0
            for j in range(n_batch):
                count = batch_counts[j]
                
                # Primary: Keep IDs
                p = batch_primaries[j]
                primary_list.append(p)
                
                # Muons: Keep IDs
                if count == 0:
                    m = [] # Empty list for no muons
                else:
                    # Extract slice
                    m = batch_muons[local_muon_idx : local_muon_idx + count]
                    local_muon_idx += count
                    m = m.tolist() # Convert inner arrays to lists for PyArrow
                
                muons_list.append(m)
            
            # Create PyArrow Table
            batch_table = pa.Table.from_pydict({
                'primary': primary_list,
                'muons': muons_list
            }, schema=schema)
            
            # Initialize writer on first batch
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, schema)
            
            writer.write_table(batch_table)
            print(f"Processed {min(i + batch_size, total_events)}/{total_events} events...")
            
        if writer:
            writer.close()
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of events per batch",
    )
    args = parser.parse_args()
    
    convert_h5_to_parquet(args.input, args.output, args.batch_size)
