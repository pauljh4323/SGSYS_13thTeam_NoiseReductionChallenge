"""BiGRU magnitude-mask inference (CPU). Weights from scripts/train.py (best.pt format)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..io import REPO_ROOT, as_waveform

_MODELS: dict = {}

FRAME = 512             # 32 ms at 16 kHz, for the "active" level measure
SILENCE_DBFS = -80.0    # at or below this measured level the input is left unscaled (g = 0)
TARGET_DBFS = {"active": -19.60, "rms": -22.03}  # train-split noisy medians (scripts/level_stats.py)


def measure_level_dbfs(x, measure: str = "active") -> float:
    """Level in dBFS: "rms" = whole-file RMS; "active" = RMS of the loudest 50% of
    non-overlapping 32 ms frames (a short tail frame is dropped unless it is the only one)."""
    x = np.asarray(x, dtype=np.float64)
    if measure == "rms":
        ms = float(np.mean(x ** 2))
    elif measure == "active":
        n = max(1, x.size // FRAME)
        e = np.mean(x[: n * FRAME].reshape(n, -1) ** 2, axis=1) if x.size >= FRAME else             np.array([np.mean(x ** 2)])
        e = np.sort(e)[::-1]
        ms = float(np.mean(e[: max(1, int(np.ceil(e.size / 2)))]))
    else:
        raise ValueError(f"level_measure must be 'active' or 'rms', got {measure!r}")
    return 10.0 * np.log10(ms) if ms > 0 else -np.inf


def level_gain_db(x, measure: str, target_dbfs: float, max_boost_db: float,
                  max_cut_db: float) -> float:
    """g = clamp(target - measured, -max_cut, +max_boost); 0 for silent / near-silent input."""
    level = measure_level_dbfs(x, measure)
    if not np.isfinite(level) or level <= SILENCE_DBFS:
        return 0.0
    return float(np.clip(target_dbfs - level, -max_cut_db, max_boost_db))


def _model(weights: str, device: str):
    path = Path(weights)
    if not path.is_absolute():  # relative to the repo root (dir of main.py), never the cwd
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


def enhance(noisy, sr: int = 16000, weights: str = "weights/gru_v1.pt", device: str = "cpu",
            level_norm: bool = False, level_measure: str = "active", target_dbfs: float | None = None,
            max_boost_db: float = 30.0, max_cut_db: float = 12.0):
    """level_norm: scale x by g = clamp(target - level) dB before the model, undo it after."""
    if sr != 16000:
        raise ValueError(f"gru_mask expects 16 kHz input, got {sr}")
    import torch
    x = as_waveform(noisy)
    g = 0.0
    if level_norm:
        if level_measure not in TARGET_DBFS:
            raise ValueError(f"level_measure must be 'active' or 'rms', got {level_measure!r}")
        target = TARGET_DBFS[level_measure] if target_dbfs is None else float(target_dbfs)
        g = level_gain_db(x, level_measure, target, max_boost_db, max_cut_db)
    scale = 10.0 ** (g / 20.0)
    model = _model(weights, device)
    with torch.inference_mode():
        y = model(torch.from_numpy((x * scale).astype(np.float32))[None].to(device))[0]
    return y.cpu().double().numpy() / scale
