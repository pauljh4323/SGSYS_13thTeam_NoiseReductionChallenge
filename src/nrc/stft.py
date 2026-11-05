"""STFT/ISTFT wrappers (scipy.signal) matching the course notebook conventions."""
from __future__ import annotations

import numpy as np
from scipy import signal

from .io import as_waveform


def check_nola(n_fft: int, hop: int, window: str = "hann") -> None:
    if not 0 < hop <= n_fft or n_fft < 4:
        raise ValueError("Require 0 < hop <= n_fft and n_fft >= 4.")
    if not signal.check_NOLA(window, n_fft, n_fft - hop):
        raise ValueError(f"STFT window/hop does not satisfy NOLA ({window}, {n_fft}, {hop}).")


def stft(x, sr: int = 16000, n_fft: int = 512, hop: int = 128, window: str = "hann"):
    """Return (freqs, times, Z). Signals shorter than n_fft are zero-padded."""
    check_nola(n_fft, hop, window)
    x = as_waveform(x)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    return signal.stft(x, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop,
                       nfft=n_fft, boundary="zeros", padded=True)


def istft(z, length: int, sr: int = 16000, n_fft: int = 512, hop: int = 128,
          window: str = "hann") -> np.ndarray:
    """Inverse STFT returning exactly `length` samples."""
    check_nola(n_fft, hop, window)
    _, y = signal.istft(z, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop,
                        nfft=n_fft, input_onesided=True, boundary=True)
    if len(y) < length:
        raise ValueError("Inverse STFT produced insufficient samples.")
    return as_waveform(y[:length])
