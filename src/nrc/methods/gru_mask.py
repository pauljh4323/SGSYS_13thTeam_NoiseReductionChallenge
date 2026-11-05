"""BiGRU magnitude-mask inference (CPU). Weights from scripts/train.py (best.pt format)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..io import REPO_ROOT, as_waveform

_MODELS: dict = {}


def _model(weights: str, device: str):
    path = Path(weights)
    if not path.is_absolute():
        path = REPO_ROOT / path
    key = (str(path.resolve()), device)
    if key not in _MODELS:
        import torch
        from ..models.gru_mask import load_weights
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        if not path.is_file():
            raise FileNotFoundError(f"gru_mask weights not found: {path}")
        _MODELS[key] = load_weights(path, device)
    return _MODELS[key]


def enhance(noisy, sr: int = 16000, weights: str = "weights/gru_v1.pt", device: str = "cpu"):
    if sr != 16000:
        raise ValueError(f"gru_mask expects 16 kHz input, got {sr}")
    import torch
    x = as_waveform(noisy)
    model = _model(weights, device)
    with torch.inference_mode():
        y = model(torch.from_numpy(x.astype(np.float32))[None].to(device))[0]
    return y.cpu().double().numpy()
