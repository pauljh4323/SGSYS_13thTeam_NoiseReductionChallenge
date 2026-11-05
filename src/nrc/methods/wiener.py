"""Faithful port of the course notebook Wiener filter (decision-directed prior SNR)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..io import as_waveform
from ..stft import check_nola, istft, stft


def estimate_noise_power(z, times, length, sr=16000, n_fft=512, noise_fraction=0.20):
    """Average the lowest-energy complete frames; no clean/reference access."""
    power = np.abs(z) ** 2
    # Avoid selecting artificially quiet frames introduced by boundary padding.
    centers = np.rint(times * sr).astype(int)
    full = np.flatnonzero((centers >= n_fft // 2) & (centers + n_fft // 2 <= length))
    if not len(full):
        full = np.arange(power.shape[1])
    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(noise_fraction * len(full))))
    selected = full[np.argsort(energy, kind="stable")[:count]]
    estimate = power[:, selected].mean(axis=1)
    return np.maximum(uniform_filter1d(estimate, size=3, mode="nearest"), 1e-18)


def validate(n_fft, hop, window, noise_fraction, wiener_smoothing, wiener_gain_floor):
    check_nola(n_fft, hop, window)
    if not 0 < noise_fraction <= 1:
        raise ValueError("noise_fraction must be in (0, 1].")
    if not 0 <= wiener_smoothing < 1:
        raise ValueError("wiener_smoothing must be in [0, 1).")
    if not 0 <= wiener_gain_floor <= 1:
        raise ValueError("wiener_gain_floor must be in [0, 1].")


def enhance(noisy, sr=16000, n_fft=512, hop=128, window="hann", noise_fraction=0.20,
            wiener_smoothing=0.98, wiener_gain_floor=0.05):
    x = as_waveform(noisy)
    if sr != 16000:
        raise ValueError("This baseline uses 16000 Hz / wideband PESQ.")
    validate(n_fft, hop, window, noise_fraction, wiener_smoothing, wiener_gain_floor)
    _, times, z = stft(x, sr, n_fft, hop, window)
    power = np.abs(z) ** 2
    noise = estimate_noise_power(z, times, len(x), sr, n_fft, noise_fraction)
    posterior_snr = power / noise[:, None]
    enhanced = np.empty_like(z)
    previous_clean_power = np.zeros_like(noise)
    for frame in range(z.shape[1]):
        instantaneous = np.maximum(posterior_snr[:, frame] - 1.0, 0.0)
        if frame == 0:
            prior_snr = instantaneous
        else:
            prior_snr = (wiener_smoothing * previous_clean_power / noise
                         + (1 - wiener_smoothing) * instantaneous)
        gain = np.clip(prior_snr / (1 + prior_snr), wiener_gain_floor, 1)
        enhanced[:, frame] = gain * z[:, frame]
        previous_clean_power = np.abs(enhanced[:, frame]) ** 2
    return istft(enhanced, len(x), sr, n_fft, hop, window)
