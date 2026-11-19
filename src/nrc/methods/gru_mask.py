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
RANGE_DBFS = {"active": (-27.88, -15.27), "rms": (-30.58, -17.25)}  # train-split p5 / p95


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


def level_gain_range_db(x, measure: str, lo_dbfs: float, hi_dbfs: float, max_boost_db: float,
                        max_cut_db: float) -> float:
    """Train-range clamp: g = 0 inside [lo, hi], lo - measured below, hi - measured above,
    then clamp(-max_cut, +max_boost); 0 for silent / near-silent input."""
    if lo_dbfs > hi_dbfs:
        raise ValueError(f"lo_dbfs ({lo_dbfs}) > hi_dbfs ({hi_dbfs})")
    level = measure_level_dbfs(x, measure)
    if not np.isfinite(level) or level <= SILENCE_DBFS:
        return 0.0
    if level < lo_dbfs:
        g = lo_dbfs - level
    elif level > hi_dbfs:
        g = hi_dbfs - level
    else:
        return 0.0
    return float(np.clip(g, -max_cut_db, max_boost_db))


def level_gain_range_boost_db(x, measure: str, lo_dbfs: float, hi_dbfs: float, target_dbfs: float,
                              max_boost_db: float, max_cut_db: float) -> float:
    """Asymmetric train-range level: g = 0 inside [lo, hi]; above hi -> hi - measured (cut to the
    edge); below lo -> target - measured (boost to the train median), then clamp(-max_cut,
    +max_boost); 0 for silent / near-silent input."""
    if not lo_dbfs <= target_dbfs <= hi_dbfs:
        raise ValueError(f"need lo_dbfs <= target_dbfs <= hi_dbfs, got {lo_dbfs}, {target_dbfs}, {hi_dbfs}")
    level = measure_level_dbfs(x, measure)
    if not np.isfinite(level) or level <= SILENCE_DBFS:
        return 0.0
    if level < lo_dbfs:
        g = target_dbfs - level
    elif level > hi_dbfs:
        g = hi_dbfs - level
    else:
        return 0.0
    return float(np.clip(g, -max_cut_db, max_boost_db))


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


# ---- mask_post: rule-based SNR-adaptive mask strength (noisy input only; exp/r2-mpost) ----
MASK_POST_DEFAULTS = {"gamma_lo": 1.0, "gamma_hi": 1.0, "snr_lo": 0.0, "snr_hi": 10.0, "floor": 0.0,
                      "highpass_hz": 0.0, "mode": "file", "smooth_s": 0.25}
SNR_EPS = 1e-12


def mask_post_params(mask_post: dict | None) -> dict | None:
    """Validate a mask_post block and fill defaults; None / empty -> None (off)."""
    if not mask_post:
        return None
    unknown = set(mask_post) - set(MASK_POST_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown mask_post keys: {sorted(unknown)}")
    p = {**MASK_POST_DEFAULTS, **mask_post}
    if p["mode"] not in ("file", "frame"):
        raise ValueError(f"mask_post mode must be 'file' or 'frame', got {p['mode']!r}")
    if p["snr_lo"] >= p["snr_hi"]:
        raise ValueError(f"mask_post needs snr_lo < snr_hi, got {p['snr_lo']}, {p['snr_hi']}")
    if min(p["gamma_lo"], p["gamma_hi"]) <= 0 or not 0.0 <= p["floor"] < 1.0:
        raise ValueError("mask_post needs gamma > 0 and 0 <= floor < 1")
    return p


def gamma_of_snr(snr_db, gamma_lo: float, gamma_hi: float, snr_lo: float, snr_hi: float):
    """gamma_lo at SNR <= snr_lo, gamma_hi at SNR >= snr_hi, linear in between (scalar or array)."""
    t = np.clip((np.asarray(snr_db, dtype=np.float64) - snr_lo) / (snr_hi - snr_lo), 0.0, 1.0)
    return gamma_lo + (gamma_hi - gamma_lo) * t


def mask_snr_db(mask, spec, hop: int = 128, sr: int = 16000, mode: str = "file", smooth_s: float = 0.25):
    """Mask-based SNR estimate from the noisy STFT only: 10 log10((sum|M X|^2 + eps) / (sum|(1-M) X|^2 + eps)).
    mask, spec: (F, T) numpy (real, complex). "file": one value; "frame": per frame (T,), with the per-frame
    numerator and denominator smoothed by a centred moving average of round(smooth_s * sr / hop) frames."""
    p = np.abs(spec).astype(np.float64) ** 2
    m = mask.astype(np.float64)
    num, den = (m ** 2 * p).sum(axis=0), ((1.0 - m) ** 2 * p).sum(axis=0)
    if mode == "file":
        return float(10.0 * np.log10((num.sum() + SNR_EPS) / (den.sum() + SNR_EPS)))
    from scipy.ndimage import uniform_filter1d
    k = max(1, int(round(smooth_s * sr / hop)))
    num, den = uniform_filter1d(num, k, mode="nearest"), uniform_filter1d(den, k, mode="nearest")
    return 10.0 * np.log10((num + SNR_EPS) / (den + SNR_EPS))


def post_mask(mask, gamma, floor: float):
    """M' = max(M^gamma, floor). mask: torch (..., F, T); gamma: scalar or (T,) numpy (per frame)."""
    import torch
    if np.ndim(gamma) == 0:
        g = float(gamma)
        out = mask if g == 1.0 else mask ** g
    else:
        out = mask ** torch.as_tensor(np.asarray(gamma), dtype=mask.dtype, device=mask.device)
    return out.clamp_min(floor) if floor > 0 else out


def highpass(y, highpass_hz: float, sr: int = 16000):
    """Same 100 Hz output high-pass as the dsp chain (mmse_lsa: Butterworth 4, sosfiltfilt)."""
    from .mmse_lsa import _highpass
    return _highpass(y, highpass_hz, sr)


def mask_and_spec(model, wave):
    """Explicit form of GRUMask.forward: returns (spec, mask, padded length). wave: torch (1, N)."""
    import torch
    n = wave.shape[-1]
    x = wave if n > model.n_fft else torch.nn.functional.pad(wave, (0, model.n_fft + 1 - n))
    spec = model.stft(x)
    return spec, model.mask(spec.abs()), x.shape[-1]


def synth(model, spec, mask, length: int, n: int):
    """iSTFT(mask * spec) exactly as GRUMask.forward, cut to n samples; torch (1, n)."""
    return model.istft(mask * spec, length=length)[..., :n]


def enhance(noisy, sr: int = 16000, weights: str = "weights/gru_v1.pt", device: str = "cpu",
            level_norm: bool = False, level_measure: str = "active", target_dbfs: float | None = None,
            max_boost_db: float = 30.0, max_cut_db: float = 12.0, level_mode: str = "target",
            lo_dbfs: float | None = None, hi_dbfs: float | None = None, mask_post: dict | None = None):
    """mask_post (optional, default off = the model's own output): rule-based mask strength from the
    noisy input only. SNR_est from the model's mask M (mask_snr_db; per file or per smoothed frame),
    gamma = gamma_of_snr(SNR_est), M' = max(M^gamma, floor), output iSTFT(M' X); then an optional
    zero-phase high-pass at highpass_hz (0 = off).
    level_norm: scale x by g dB before the model, undo it after.
    level_mode "target": g = clamp(target - level); "range": g moves the level only as far as the
    nearest edge of [lo_dbfs, hi_dbfs] (g = 0 inside it); "range_boost": as "range" above hi, but
    below lo boosts to target_dbfs (train median). All clamped by max_boost/max_cut."""
    if sr != 16000:
        raise ValueError(f"gru_mask expects 16 kHz input, got {sr}")
    import torch
    x = as_waveform(noisy)
    g = 0.0
    if level_norm:
        if level_measure not in TARGET_DBFS:
            raise ValueError(f"level_measure must be 'active' or 'rms', got {level_measure!r}")
        if level_mode == "target":
            target = TARGET_DBFS[level_measure] if target_dbfs is None else float(target_dbfs)
            g = level_gain_db(x, level_measure, target, max_boost_db, max_cut_db)
        elif level_mode == "range":
            lo = RANGE_DBFS[level_measure][0] if lo_dbfs is None else float(lo_dbfs)
            hi = RANGE_DBFS[level_measure][1] if hi_dbfs is None else float(hi_dbfs)
            g = level_gain_range_db(x, level_measure, lo, hi, max_boost_db, max_cut_db)
        elif level_mode == "range_boost":
            lo = RANGE_DBFS[level_measure][0] if lo_dbfs is None else float(lo_dbfs)
            hi = RANGE_DBFS[level_measure][1] if hi_dbfs is None else float(hi_dbfs)
            target = TARGET_DBFS[level_measure] if target_dbfs is None else float(target_dbfs)
            g = level_gain_range_boost_db(x, level_measure, lo, hi, target, max_boost_db, max_cut_db)
        else:
            raise ValueError(f"level_mode must be 'target', 'range' or 'range_boost', got {level_mode!r}")
    scale = 10.0 ** (g / 20.0)  # g = 0 -> scale exactly 1.0 -> bit-identical to level_norm off
    mp = mask_post_params(mask_post)
    model = _model(weights, device)
    with torch.inference_mode():
        wave = torch.from_numpy((x * scale).astype(np.float32))[None].to(device)
        if mp is None:
            y = model(wave)[0]
        else:
            spec, mask, length = mask_and_spec(model, wave)
            snr = mask_snr_db(mask[0].cpu().numpy(), spec[0].cpu().numpy(), model.hop, sr,
                              mp["mode"], mp["smooth_s"])
            gamma = gamma_of_snr(snr, mp["gamma_lo"], mp["gamma_hi"], mp["snr_lo"], mp["snr_hi"])
            y = synth(model, spec, post_mask(mask, gamma, mp["floor"]), length, wave.shape[-1])[0]
    y = y.cpu().double().numpy() / scale
    return highpass(y, mp["highpass_hz"], sr) if mp is not None else y
