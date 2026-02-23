"""
utils/device.py
Detects the best available compute device.
Mac = MPS (Apple Silicon) or CPU. Never CUDA.
"""

import torch
import os
from dotenv import load_dotenv

load_dotenv()


def get_device() -> torch.device:
    """
    Returns the best available device based on environment config.
    Priority: CUDA → MPS → CPU
    Override via DEVICE= in .env (auto | cpu | mps | cuda)
    """
    config = os.getenv("DEVICE", "auto").lower()

    if config == "cpu":
        return torch.device("cpu")

    if config == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    if config == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("[WARN] MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    # auto mode - pick best available
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
