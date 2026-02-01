import torch
import torch.nn as nn
import torch.autograd as autograd
import sys
import traceback

# Ensure we can import from local directory
sys.path.append('.')

try:
    from model import ScalableCritic, compute_gp_flat
except ImportError:
    # If running from parent dir, try appending 'training'
    sys.path.append('training')
    from model import ScalableCritic, compute_gp_flat

def test_gp_crash():
    print("Starting test_gp_crash...", flush=True)
    device = torch.device("cpu")
    feat_dim = 3
    cond_dim = 4
    batch_size = 2
    
    print("Initializing Critic...", flush=True)
    # Initialize Critic
    crit = ScalableCritic(feat_dim=feat_dim, cond_dim=cond_dim, device=device)
    
    # Create dummy data
    real_muons = torch.randn(5, feat_dim, device=device)
    fake_muons = torch.randn(5, feat_dim, device=device)
    batch_idx = torch.tensor([0, 0, 1, 1, 1], device=device)
    conditions = torch.randn(batch_size, cond_dim, device=device)
    
    print("Testing compute_gp_flat...", flush=True)
    try:
        gp = compute_gp_flat(
            crit, 
            real_muons, 
            fake_muons, 
            batch_idx, 
            conditions, 
            batch_size, 
            device=device
        )
        print(f"Gradient Penalty computed: {gp.item()}", flush=True)
        
        # Simulate loss backward which triggers double backward through GP
        loss = gp * 10.0
        print("Calling backward on GP loss...", flush=True)
        loss.backward()
        print("Backward successful! Fix is working.", flush=True)
        
    except RuntimeError as e:
        if "Trying to backward through the graph a second time" in str(e):
            print("\n!!! REPRODUCED ISSUE: RuntimeError caught !!!", flush=True)
        else:
            print(f"\nCaught unexpected RuntimeError: {e}", flush=True)
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"\nCaught unexpected exception: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_gp_crash()
    except Exception:
        traceback.print_exc()
