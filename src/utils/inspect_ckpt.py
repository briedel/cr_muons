import torch
import sys
import os

ckpt_path = "logs_tensorboard/20260202-175404/checkpoints/last.ckpt" 

# Find the latest checkpoint systematically
import glob
files = glob.glob("logs_tensorboard/**/checkpoints/last.ckpt", recursive=True)
if not files:
    print("No checkpoints found.")
    sys.exit(1)

latest_ckpt = max(files, key=os.path.getmtime)
print(f"Inspecting: {latest_ckpt}")

try:
    ckpt = torch.load(latest_ckpt, map_location="cpu")
    print(f"Checkpoint Epoch: {ckpt.get('epoch', 'Unknown')}")
    print(f"Checkpoint Global Step: {ckpt.get('global_step', 'Unknown')}")
    if "datamodule_state_dict" in ckpt:
        dm_state = ckpt["datamodule_state_dict"]
        print("DataModule State Keys:", dm_state.keys())
        if "processed_files" in dm_state:
            pf = dm_state["processed_files"]
            print(f"Processed Files Count: {len(pf)}")
            print("First 5:", pf[:5])
        else:
            print("processed_files key MISSING in datamodule_state_dict")
    else:
        print("datamodule_state_dict MISSING in checkpoint")
        print("Keys:", ckpt.keys())
        if "MuonDataModule" in ckpt:
             print("Found 'MuonDataModule' key. Content:", ckpt["MuonDataModule"])

except Exception as e:
    print(f"Error loading: {e}")
