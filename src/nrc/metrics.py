"""Evaluation metrics. Course SNR/PESQ definitions are ported from the course notebook."""
from __future__ import annotations

import numpy as np

from .io import as_waveform


def snr_db(clean, estimate) -> float:
    """Course SNR: 10*log10(sum(clean^2) / sum((clean-estimate)^2)), no level alignment."""
    clean, estimate = as_waveform(clean), as_waveform(estimate)
    if clean.shape != estimate.shape:
        raise ValueError("SNR requires equal lengths.")
    energy = np.sum(clean ** 2)
    if energy <= 0:
        raise ValueError("SNR is undefined for a silent reference.")
    error = np.sum((clean - estimate) ** 2)
    return float(np.inf if error == 0 else 10 * np.log10(energy / error))


def pesq_wb(clean, estimate, sample_rate=16000) -> tuple[float, str]:
    """Wideband PESQ; returns (nan, error message) on failure instead of raising."""
    from pesq import PesqError, pesq
    try:
        value = float(pesq(sample_rate, clean, estimate, mode="wb",
                           on_error=PesqError.RAISE_EXCEPTION))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid PESQ return value: {value}")
        return value, ""
    except Exception as error:  # noqa: BLE001 - counted as failed, never dropped
        return float("nan"), f"{type(error).__name__}: {error}"


def si_sdr(clean, estimate, eps=1e-12) -> float:
    """Scale-invariant SDR (Le Roux et al. 2019), zero-mean signals."""
    s = as_waveform(clean) - np.mean(clean)
    y = as_waveform(estimate) - np.mean(estimate)
    alpha = np.dot(y, s) / (np.dot(s, s) + eps)
    target = alpha * s
    noise = y - target
    return float(10 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def stoi(clean, estimate, sample_rate=16000) -> float:
    from pystoi import stoi as _stoi
    return float(_stoi(clean, estimate, sample_rate, extended=False))
