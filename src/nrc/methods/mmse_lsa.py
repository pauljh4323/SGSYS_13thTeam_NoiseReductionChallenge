"""Ephraim-Malah MMSE log-spectral amplitude (LSA) estimator.

Decision-directed a priori SNR; noise PSD from either the lowest-energy frames
("initial", same as the Wiener port) or Cohen's MCRA (minima-controlled recursive averaging).
"""
from __future__ import annotations

import numpy as np
from scipy.special import exp1

from ..io import as_waveform
from ..stft import check_nola, istft, stft
from .wiener import estimate_noise_power

_EPS = 1e-18
_V_MIN, _V_MAX = 1e-8, 500.0  # exp1 finite and exp(0.5*exp1(v)) bounded in this range


def lsa_gain(xi, gamma):
    """G = xi/(1+xi) * exp(0.5 * E1(v)), v = xi*gamma/(1+xi)."""
    v = np.clip(xi * gamma / (1.0 + xi), _V_MIN, _V_MAX)
    return xi / (1.0 + xi) * np.exp(0.5 * exp1(v))


def mcra_noise(power, alpha_d=0.95, alpha_p=0.2, alpha_s=0.8, window=96, delta=5.0):
    """Cohen (2002) MCRA noise PSD per frame; power is |Y|^2 of shape (bins, frames)."""
    n_bins, n_frames = power.shape
    b = np.array([0.25, 0.5, 0.25])
    noise = np.empty_like(power)
    s = s_min = s_tmp = lam = None
    p = np.zeros(n_bins)
    for t in range(n_frames):
        y2 = power[:, t]
        sf = np.convolve(np.pad(y2, 1, mode="edge"), b, mode="valid")
        if t == 0:
            s = sf.copy()
            s_min = sf.copy()
            s_tmp = sf.copy()
            lam = np.maximum(y2, _EPS)
        else:
            s = alpha_s * s + (1 - alpha_s) * sf
            s_min = np.minimum(s_min, s)
            s_tmp = np.minimum(s_tmp, s)
            if t % window == 0:
                s_min = np.minimum(s_tmp, s)
                s_tmp = s.copy()
            speech = (s > delta * np.maximum(s_min, _EPS)).astype(float)
            p = alpha_p * p + (1 - alpha_p) * speech
            a = alpha_d + (1 - alpha_d) * p
            lam = np.maximum(a * lam + (1 - a) * y2, _EPS)
        noise[:, t] = lam
    return noise


def validate(n_fft, hop, window, noise_est, noise_fraction, alpha, alpha_d, alpha_p, alpha_s,
             mcra_window, delta):
    check_nola(n_fft, hop, window)
    if noise_est not in ("initial", "mcra"):
        raise ValueError("noise_est must be 'initial' or 'mcra'.")
    if not 0 < noise_fraction <= 1:
        raise ValueError("noise_fraction must be in (0, 1].")
    for name, v in (("alpha", alpha), ("alpha_d", alpha_d), ("alpha_p", alpha_p),
                    ("alpha_s", alpha_s)):
        if not 0 <= v < 1:
            raise ValueError(f"{name} must be in [0, 1).")
    if mcra_window < 1 or delta <= 0:
        raise ValueError("mcra_window must be >= 1 and delta > 0.")


def enhance(noisy, sr=16000, n_fft=512, hop=128, window="hann", noise_est="initial",
            noise_fraction=0.20, alpha=0.98, xi_min_db=-25.0, gmin_db=-25.0,
            alpha_d=0.95, alpha_p=0.2, alpha_s=0.8, mcra_window=96, delta=5.0):
    x = as_waveform(noisy)
    if sr != 16000:
        raise ValueError("This method uses 16000 Hz / wideband PESQ.")
    validate(n_fft, hop, window, noise_est, noise_fraction, alpha, alpha_d, alpha_p, alpha_s,
             mcra_window, delta)
    _, times, z = stft(x, sr, n_fft, hop, window)
    power = np.abs(z) ** 2
    if noise_est == "initial":
        noise = np.repeat(estimate_noise_power(z, times, len(x), sr, n_fft, noise_fraction)[:, None],
                          z.shape[1], axis=1)
    else:
        noise = mcra_noise(power, alpha_d, alpha_p, alpha_s, int(mcra_window), delta)
    noise = np.maximum(noise, _EPS)
    xi_min = 10.0 ** (xi_min_db / 10.0)
    gmin = 10.0 ** (gmin_db / 20.0)
    gamma = np.minimum(power / noise, 1e8)
    enhanced = np.empty_like(z)
    prev_amp2 = np.zeros(z.shape[0])
    for t in range(z.shape[1]):
        inst = np.maximum(gamma[:, t] - 1.0, 0.0)
        if t == 0:
            xi = alpha + (1 - alpha) * inst  # standard DD initialisation
        else:
            xi = alpha * prev_amp2 / noise[:, t] + (1 - alpha) * inst
        xi = np.maximum(xi, xi_min)
        g = np.clip(lsa_gain(xi, gamma[:, t]), gmin, 1.0)
        enhanced[:, t] = g * z[:, t]
        prev_amp2 = np.abs(enhanced[:, t]) ** 2
    return istft(enhanced, len(x), sr, n_fft, hop, window)
