"""Checkpoint and model I/O utilities for training persistence."""

import json
import os
import tempfile

import torch


def fs_put_json(fs, remote_path: str, data: dict) -> None:
    """Upload JSON data to a remote filesystem."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            json.dump(data, tmp)
        fs.put(tmp_path, remote_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def fs_put_torch_checkpoint(fs, remote_path: str, checkpoint_data: dict) -> None:
    """Upload PyTorch checkpoint to a remote filesystem."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(checkpoint_data, tmp_path)
        fs.put(tmp_path, remote_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def fs_put_file(fs, remote_path: str, local_path: str) -> None:
    """Upload a local file to a remote path via fs.put."""
    fs.put(str(local_path), str(remote_path))


def load_progress(checkpoint_path, fs=None) -> tuple[int, set[str]]:
    """Load progress tracking.

    Backward compatible:
      - Old format: {"processed_files": [...]} (implicit epoch=0)
      - New format: {"epoch": int, "processed_files": [...]} (epoch is the current epoch)
    """
    processed_files: set[str] = set()
    epoch = 0

    def _parse(data: object) -> None:
        nonlocal epoch, processed_files
        if isinstance(data, dict):
            epoch = int(data.get("epoch", 0) or 0)
            processed_files = set(data.get("processed_files", []) or [])

    if fs:
        try:
            if fs.exists(checkpoint_path):
                with fs.open(checkpoint_path, 'r') as f:
                    _parse(json.load(f))
        except Exception as e:
            print(f"Warning: Could not read checkpoint from Pelican: {e}")
    elif checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            try:
                _parse(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode checkpoint file {checkpoint_path}")

    return epoch, processed_files


def save_progress(checkpoint_path, epoch: int, processed_files: set[str], fs=None) -> None:
    """Save progress tracking."""
    if not checkpoint_path:
        return
    payload = {"epoch": int(epoch), "processed_files": list(processed_files)}
    if fs:
        try:
            fs_put_json(fs, checkpoint_path, payload)
        except Exception as e:
            print(f"Warning: Could not save checkpoint to Pelican: {e}")
    else:
        with open(checkpoint_path, 'w') as f:
            json.dump(payload, f)


def save_model_checkpoint(path, gen, crit, opt_G, opt_C, epoch=0, fs=None):
    """Save model and optimizer states."""
    checkpoint_data = {
        'gen_state_dict': gen.state_dict(),
        'crit_state_dict': crit.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_C_state_dict': opt_C.state_dict(),
        'epoch': epoch
    }
    
    if fs:
        try:
            fs_put_torch_checkpoint(fs, path, checkpoint_data)
            print(f"Model checkpoint saved to Pelican: {path}")
        except Exception as e:
            print(f"Warning: Could not save model checkpoint to Pelican: {e}")
    else:
        torch.save(checkpoint_data, path)
        print(f"Model checkpoint saved to {path}")


def load_model_checkpoint(path, gen, crit, opt_G, opt_C, device, fs=None):
    """Load model and optimizer states."""
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
        # Try to load optimizer state, but skip if it fails (e.g., optimizer type mismatch)
        try:
            opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
            opt_C.load_state_dict(checkpoint['opt_C_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state (may be optimizer type mismatch): {e}")
            print("       Optimizer state reset; training will continue with fresh optimizer.")
        return checkpoint.get('epoch', 0)
    return 0
