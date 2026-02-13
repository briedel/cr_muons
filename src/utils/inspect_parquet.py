import pyarrow.parquet as pq
import pyarrow as pa
import argparse
import glob
import os
import numpy as np

def inspect_files(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"No files found matching: {pattern}")
        return

    print(f"{'Filename':<50} | {'Total':<10} | {'With Muons':<10} | {'% Non-Empty':<10}")
    print("-" * 90)

    grand_total_rows = 0
    grand_total_muons = 0

    for fpath in files:
        try:
            pf = pq.ParquetFile(fpath)
            num_rows = pf.metadata.num_rows

            # Print first 10 primaries for the first file to inspect data
            if fpath == files[0]:
                try:
                    print(f"\n[First 10 Events from {os.path.basename(fpath)}]")
                    # Try to read just the first 10 rows
                    head_batch = next(pf.iter_batches(batch_size=10))
                    
                    prims = None
                    if 'primary' in head_batch.column_names:
                         prims = head_batch['primary']
                    elif 'primaries' in head_batch.column_names:
                         prims = head_batch['primaries']
                         
                    muons = None
                    if 'muons' in head_batch.column_names:
                        muons = head_batch['muons']
                    
                    if prims:
                        for i in range(len(prims)):
                            val = prims[i].as_py()
                            # Formatting for cleaner output if it's a list/array
                            if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
                                val_str = ", ".join([f"{x:.4g}" if isinstance(x, float) else str(x) for x in val])
                                print(f"  {i}: Primary: [{val_str}]")
                            else:
                                print(f"  {i}: Primary: {val}")
                            
                            if muons:
                                mu_val = muons[i].as_py()
                                if hasattr(mu_val, '__iter__'):
                                    if len(mu_val) == 0:
                                        print(f"     Muons: []")
                                    else:
                                        print(f"     Muons ({len(mu_val)}):")
                                        # Print max 5 muons to avoid spam
                                        for k, m in enumerate(mu_val[:5]):
                                            if hasattr(m, '__iter__'):
                                                 m_str = ", ".join([f"{x:.4g}" if isinstance(x, float) else str(x) for x in m])
                                                 print(f"       - [{m_str}]")
                                            else:
                                                 print(f"       - {m}")
                                        if len(mu_val) > 5:
                                            print(f"       ... and {len(mu_val)-5} more")
                                else:
                                    print(f"     Muons: {mu_val}")

                    else:
                        print(f"  Column 'primary'/'primaries' not found. Columns: {head_batch.column_names}")
                    print("-" * 50 + "\n")
                except Exception as e:
                    print(f"  [Could not read primaries: {e}]\n")
            
            # To count non-empty efficiently, we only read the 'muons' column
            # and look at the offsets or lengths.
            # We'll read in batches to avoid memory issues for very large files.
            with_muons_count = 0
            for batch in pf.iter_batches(batch_size=10000, columns=["muons"]):
                mu_col = batch.column(0)
                
                # Try to use offsets if it's a list type
                if hasattr(mu_col, 'offsets'):
                    offsets = mu_col.offsets.to_numpy()
                    counts = np.diff(offsets)
                    with_muons_count += np.count_nonzero(counts > 0)
                else:
                    # Fallback for other types
                    for i in range(len(mu_col)):
                        val = mu_col[i].as_py()
                        if val is not None and len(val) > 0:
                            with_muons_count += 1
            
            percentage = (with_muons_count / num_rows * 100) if num_rows > 0 else 0
            fname = os.path.basename(fpath)
            print(f"{fname[:50]:<50} | {num_rows:<10} | {with_muons_count:<10} | {percentage:>9.1f}%")
            
            grand_total_rows += num_rows
            grand_total_muons += with_muons_count
            
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    print("-" * 90)
    grand_percentage = (grand_total_muons / grand_total_rows * 100) if grand_total_rows > 0 else 0
    print(f"{'TOTAL':<50} | {grand_total_rows:<10} | {grand_total_muons:<10} | {grand_percentage:>9.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Parquet files for muon event counts.")
    parser.add_argument("path", help="Path or glob pattern to Parquet files")
    args = parser.parse_args()
    
    inspect_files(args.path)
