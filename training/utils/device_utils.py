"""PyTorch device selection and detection utilities."""

import torch


def select_torch_device(device_arg: str) -> torch.device:
    """Select torch device from CLI arg.

    device_arg:
      - "auto": prefer CUDA, then MPS, else CPU
      - "cuda": require CUDA
      - "mps": require Apple Metal (PyTorch MPS)
      - "cpu": force CPU
    """
    device_arg = (device_arg or "auto").lower()

    has_mps_backend = bool(getattr(torch.backends, "mps", None))
    mps_available = bool(torch.backends.mps.is_available()) if has_mps_backend else False
    rocm_build = bool(getattr(torch.version, "hip", None))

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    if device_arg == "rocm":
        # PyTorch uses the 'cuda' device type for ROCm as well.
        if not rocm_build:
            raise RuntimeError("--device rocm was requested, but this PyTorch build is not ROCm-enabled")
        if not torch.cuda.is_available():
            raise RuntimeError("--device rocm was requested, but no ROCm device is available")
        return torch.device("cuda")

    if device_arg == "mps":
        if not mps_available:
            raise RuntimeError("--device mps was requested, but MPS is not available")
        return torch.device("mps")

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError(
        f"Unknown --device value: {device_arg}. Use one of: auto, cpu, cuda, rocm, mps"
    )


def device_backend_label(device: torch.device) -> str:
    """Get a human-readable label for the device backend."""
    if device.type == "cuda":
        # CUDA device type may mean NVIDIA CUDA or AMD ROCm.
        if bool(getattr(torch.version, "hip", None)):
            return "rocm"
        return "cuda"
    return device.type
