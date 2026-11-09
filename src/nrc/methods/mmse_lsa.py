"""Ephraim-Malah MMSE log-spectral amplitude (LSA) estimator.

Decision-directed a priori SNR; noise PSD from one of:
  "initial"  the lowest-energy frames (same as the Wiener port),
  "mcra"     Cohen's MCRA (minima-controlled recursive averaging, causal),
  "quantile" per-bin q-th quantile of the noisy power over the whole file, times a bias.
             NON-CAUSAL: it looks at every frame of the file. That is allowed here because
             inference is offline, file-by-file (the full noisy file is available up front).

Optional extensions (defaults reproduce the plain DD/LSA path bit-identically):
  noise_bias             multiply the noise PSD estimate by this factor (over/under-estimation).
  xi_cepstral_smoothing  first-step a priori SNR from selective cepstro-temporal smoothing of the
                         ML speech PSD (Breithaupt, Gerkmann, Martin 2008) instead of DD.
  xi_mode "tsnr"         two-step noise reduction (Plapous, Marro, Scalart 2006):
                         xi2 = |G1 * Y|^2 / lambda_d, final gain from xi2.
  xi_mode "hrnr"         TSNR + harmonic regeneration: half-wave rectified time-domain TSNR output,
                         re-STFT, xi = (rho |G_tsnr Y|^2 + (1-rho) |S_harmo|^2) / lambda_d.
                         Non-causal over the file (offline processing).
  highpass_hz            > 0: Butterworth high-pass (designed order 4) applied forward-backward
                         (sosfiltfilt, zero-phase, effective magnitude order 8) to the output.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.special import exp1

from ..io import as_waveform
from ..stft import check_nola, istft, stft
from .wiener import estimate_noise_power

_EPS = 1e-18
_V_MIN, _V_MAX = 1e-8, 500.0  # exp1 finite and exp(0.5*exp1(v)) bounded in this range
_EULER = 0.5772156649015329
XI_MODES = ("dd", "tsnr", "hrnr")


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


def quantile_noise(power, q=0.2, bias=1.0, smooth_frames=0):
    """Stationary noise PSD (bins,): bias * q-th quantile over frames of |Y|^2.

    smooth_frames > 1 first applies a centred moving average of that many frames along time
    (reduces the variance of the per-frame periodogram before taking the quantile).
    Non-causal: uses all frames of the file.
    """
    p = power
    k = int(smooth_frames)
    if k > 1 and p.shape[1] > 1:
        k = min(k, p.shape[1])
        kernel = np.ones(k) / k
        padded = np.pad(p, ((0, 0), (k // 2, k - 1 - k // 2)), mode="edge")
        p = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)
    return bias * np.quantile(p, q, axis=1)


def cepstral_xi(gamma, noise, xi_min, sr=16000, alpha_env=0.5, alpha_high=0.97,
                alpha_pitch=0.4, env_ms=0.3, f0_min=70.0, f0_max=500.0):
    """Selective cepstro-temporal smoothing of the ML speech PSD (Breithaupt et al. 2008).

    ML PSD = lambda_d * max(gamma - 1, xi_min); its real cepstrum is smoothed recursively over
    frames, c_s(t) = a(q) c_s(t-1) + (1 - a(q)) c(t), with weak smoothing (alpha_env) for the
    envelope quefrencies (q <= env_ms), alpha_pitch at the per-frame pitch peak (+-1 quefrency bin,
    searched in sr/f0_max..sr/f0_min) and strong smoothing (alpha_high) elsewhere. The log-domain
    bias of an exponentially distributed periodogram (-Euler gamma) is compensated by exp(Euler).
    Returns xi of shape (bins, frames).
    """
    n_bins, n_frames = gamma.shape
    n = 2 * (n_bins - 1)
    log_p = np.log(np.maximum(noise * np.maximum(gamma - 1.0, xi_min), _EPS))
    ceps = np.fft.irfft(log_p, n=n, axis=0)  # real, even
    q = np.minimum(np.arange(n), n - np.arange(n))
    base = np.where(q <= int(round(env_ms * 1e-3 * sr)), alpha_env, alpha_high)
    lo, hi = int(np.floor(sr / f0_max)), min(int(np.ceil(sr / f0_min)), n // 2 - 1)
    smoothed = np.empty_like(ceps)
    c_s = ceps[:, 0].copy()
    for t in range(n_frames):
        c = ceps[:, t]
        a = base.copy()
        if hi > lo:
            k = lo + int(np.argmax(c[lo:hi + 1]))
            for j in (k - 1, k, k + 1):
                a[j] = a[n - j] = alpha_pitch
        if t > 0:
            c_s = a * c_s + (1.0 - a) * c
        smoothed[:, t] = c_s
    psd = np.exp(np.fft.rfft(smoothed, axis=0).real + _EULER)
    return np.maximum(psd / noise, xi_min)


def validate(n_fft, hop, window, noise_est, noise_fraction, alpha, alpha_d, alpha_p, alpha_s,
             mcra_window, delta, q=0.2, bias=1.0, smooth_frames=0, noise_bias=1.0,
             xi_mode="dd", hrnr_rho=0.5, highpass_hz=0.0, sr=16000):
    check_nola(n_fft, hop, window)
    if noise_bias <= 0:
        raise ValueError("noise_bias must be > 0.")
    if xi_mode not in XI_MODES:
        raise ValueError(f"xi_mode must be one of {XI_MODES}.")
    if not 0 <= hrnr_rho <= 1:
        raise ValueError("hrnr_rho must be in [0, 1].")
    if not (highpass_hz == 0 or 0 < highpass_hz < sr / 2):
        raise ValueError("highpass_hz must be 0 (off) or in (0, sr/2).")
    if noise_est not in ("initial", "mcra", "quantile"):
        raise ValueError("noise_est must be 'initial', 'mcra' or 'quantile'.")
    if not 0 <= q <= 1 or bias <= 0 or smooth_frames < 0:
        raise ValueError("quantile needs q in [0, 1], bias > 0, smooth_frames >= 0.")
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
            alpha_d=0.95, alpha_p=0.2, alpha_s=0.8, mcra_window=96, delta=5.0,
            q=0.2, bias=1.0, smooth_frames=0, noise_bias=1.0, xi_mode="dd",
            xi_cepstral_smoothing=False, hrnr_rho=0.5, highpass_hz=0.0):
    x = as_waveform(noisy)
    if sr != 16000:
        raise ValueError("This method uses 16000 Hz / wideband PESQ.")
    validate(n_fft, hop, window, noise_est, noise_fraction, alpha, alpha_d, alpha_p, alpha_s,
             mcra_window, delta, q, bias, smooth_frames, noise_bias, xi_mode, hrnr_rho,
             highpass_hz, sr)
    _, times, z = stft(x, sr, n_fft, hop, window)
    power = np.abs(z) ** 2
    if noise_est == "initial":
        noise = np.repeat(estimate_noise_power(z, times, len(x), sr, n_fft, noise_fraction)[:, None],
                          z.shape[1], axis=1)
    elif noise_est == "quantile":
        noise = np.repeat(quantile_noise(power, q, bias, smooth_frames)[:, None], z.shape[1], axis=1)
    else:
        noise = mcra_noise(power, alpha_d, alpha_p, alpha_s, int(mcra_window), delta)
    if noise_bias != 1.0:
        noise = noise * noise_bias
    noise = np.maximum(noise, _EPS)
    xi_min = 10.0 ** (xi_min_db / 10.0)
    gmin = 10.0 ** (gmin_db / 20.0)
    gamma = np.minimum(power / noise, 1e8)
    if xi_mode != "dd" or xi_cepstral_smoothing:
        y = _enhance_two_step(z, gamma, noise, alpha, xi_min, gmin, xi_mode,
                              bool(xi_cepstral_smoothing), hrnr_rho, len(x), sr, n_fft, hop, window)
    else:
        y = _enhance_dd(z, gamma, noise, alpha, xi_min, gmin, len(x), sr, n_fft, hop, window)
    return _highpass(y, highpass_hz, sr)


def _highpass(y, highpass_hz, sr):
    if not highpass_hz:
        return y
    sos = signal.butter(4, highpass_hz, btype="highpass", fs=sr, output="sos")
    if len(y) <= 3 * (2 * len(sos) + 1):  # too short for sosfiltfilt's default padding
        return y
    return signal.sosfiltfilt(sos, y)


def _enhance_dd(z, gamma, noise, alpha, xi_min, gmin, length, sr, n_fft, hop, window):
    """Plain decision-directed LSA (the dsp-v1 path, unchanged)."""
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
    return istft(enhanced, length, sr, n_fft, hop, window)


def _enhance_two_step(z, gamma, noise, alpha, xi_min, gmin, xi_mode, cepstral, rho, length, sr,
                      n_fft, hop, window):
    """First-step xi (DD or cepstral), then optional TSNR / HRNR second step."""
    xi_cs = cepstral_xi(gamma, noise, xi_min, sr) if cepstral else None
    g_final = np.empty(z.shape)
    prev_amp2 = np.zeros(z.shape[0])
    for t in range(z.shape[1]):
        gm = gamma[:, t]
        if cepstral:
            xi1 = xi_cs[:, t]
        else:
            inst = np.maximum(gm - 1.0, 0.0)
            xi1 = (alpha + (1 - alpha) * inst if t == 0
                   else alpha * prev_amp2 / noise[:, t] + (1 - alpha) * inst)
            xi1 = np.maximum(xi1, xi_min)
        g = np.clip(lsa_gain(xi1, gm), gmin, 1.0)
        if xi_mode != "dd":  # TSNR second step: xi2 = |G1 Y|^2 / lambda_d = G1^2 gamma
            xi2 = np.maximum(g * g * gm, xi_min)
            g = np.clip(lsa_gain(xi2, gm), gmin, 1.0)
        g_final[:, t] = g
        prev_amp2 = g * g * gm * noise[:, t]  # |S_hat|^2 of the final estimate (DD recursion)
    if xi_mode == "hrnr":
        s_tsnr = istft(g_final * z, length, sr, n_fft, hop, window)
        _, _, z_h = stft(np.maximum(s_tsnr, 0.0), sr, n_fft, hop, window)
        z_h = z_h[:, :z.shape[1]]
        xi_h = (rho * g_final ** 2 * np.abs(z) ** 2 + (1 - rho) * np.abs(z_h) ** 2) / noise
        g_final = np.clip(lsa_gain(np.maximum(xi_h, xi_min), gamma), gmin, 1.0)
    return istft(g_final * z, length, sr, n_fft, hop, window)
