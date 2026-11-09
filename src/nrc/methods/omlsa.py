"""IMCRA noise estimation and OM-LSA gain (Cohen 2003; Cohen & Berdugo 2001).

noise_est:
  "initial"  lowest-energy frames (same estimate as mmse_lsa / the Wiener port), constant in time,
  "imcra"    Cohen (2003) improved minima-controlled recursive averaging: smoothing + minimum
             tracking over U sub-windows of V frames, rough speech-presence gate, a second
             smoothing/minimum-tracking pass on the gated power, a priori absence q from that pass,
             speech presence p, p-controlled recursive averaging, bias compensation beta.
gain_mode:
  "lsa"      G = max(G_LSA, G_min)            (identical to mmse_lsa)
  "omlsa"    G = G_LSA^p * G_min^(1-p)         p = speech presence probability with the a priori
             absence q of Cohen & Berdugo (2001) (local / global / frame recursive-xi measures),
             capped at q_max. As in that paper the decision-directed xi uses G_LSA (gain under H1).
Only the noisy waveform and the sample rate are used.
"""
from __future__ import annotations

import numpy as np

from ..io import as_waveform
from ..stft import check_nola, istft, stft
from .mmse_lsa import lsa_gain
from .wiener import estimate_noise_power

_EPS = 1e-18


def _hann_kernel(w: int) -> np.ndarray:
    """Normalised MATLAB-style hanning(2w+1) (no zero end points)."""
    h = np.hanning(2 * w + 3)[1:-1]
    return h / h.sum()


def _smooth(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Frequency smoothing with edge padding (keeps the level at the band edges)."""
    w = len(kernel) // 2
    if not w:
        return x.copy()
    padded = np.concatenate((np.full(w, x[0]), x, np.full(w, x[-1])))
    return np.convolve(padded, kernel, mode="valid")


def _presence(q, xi, gamma):
    """p = 1 / (1 + q/(1-q) (1+xi) exp(-v)), v = gamma xi / (1+xi); q in [0, 1]."""
    v = np.minimum(gamma * xi / (1.0 + xi), 500.0)
    ratio = q / np.maximum(1.0 - q, 1e-12) * (1.0 + xi) * np.exp(-v)  # finite: q, xi bounded
    return 1.0 / (1.0 + ratio)


def _log_map(z, lo, hi):
    """0 below lo, 1 above hi, log-linear in between (Cohen & Berdugo 2001, eq. for P_local)."""
    return np.clip(np.log(np.maximum(z, _EPS) / lo) / np.log(hi / lo), 0.0, 1.0)


class IMCRA:
    """Cohen (2003) IMCRA, one frame at a time. Paper defaults for all constants."""

    def __init__(self, init_power, alpha_s=0.9, alpha_d=0.85, beta=1.47, w=1, U=8, V=15,
                 b_min=1.66, gamma0=4.6, gamma1=3.0, zeta0=1.67):
        self.alpha_s, self.alpha_d, self.beta = alpha_s, alpha_d, beta
        self.U, self.V, self.b_min = int(U), int(V), b_min
        self.gamma0, self.gamma1, self.zeta0 = gamma0, gamma1, zeta0
        self.kernel = _hann_kernel(int(w))
        y0 = np.maximum(init_power, _EPS)
        s0 = _smooth(y0, self.kernel)
        self.S = s0.copy()
        self.S_t = s0.copy()
        self.s_min, self.s_tmp = s0.copy(), s0.copy()
        self.s_min_t, self.s_tmp_t = s0.copy(), s0.copy()
        self.buf = np.repeat(s0[None], self.U, axis=0)      # stored sub-window minima, 1st pass
        self.buf_t = np.repeat(s0[None], self.U, axis=0)    # 2nd pass
        self.count = 0
        self.lam_bar = y0.copy()
        self.lam = beta * self.lam_bar

    def update(self, y2, xi, gamma):
        """Advance with frame power y2 and that frame's xi/gamma; returns lambda_d for the next frame."""
        a_s, bm = self.alpha_s, self.b_min
        # 1st iteration: smoothing and minimum tracking
        self.S = a_s * self.S + (1 - a_s) * _smooth(y2, self.kernel)
        self.s_min = np.minimum(self.s_min, self.S)
        self.s_tmp = np.minimum(self.s_tmp, self.S)
        # rough speech-presence decision
        smin = np.maximum(self.s_min, _EPS)
        gmin = y2 / (bm * smin)
        zeta = self.S / (bm * smin)
        ind = ((gmin < self.gamma0) & (zeta < self.zeta0)).astype(float)
        # 2nd iteration: smoothing restricted to (roughly) speech-absent bins
        num = np.convolve(ind * y2, self.kernel, mode="same")
        den = np.convolve(ind, self.kernel, mode="same")
        has = den > 0
        sf_t = np.where(has, num / np.where(has, den, 1.0), self.S_t)
        self.S_t = np.where(has, a_s * self.S_t + (1 - a_s) * sf_t, self.S_t)
        self.s_min_t = np.minimum(self.s_min_t, self.S_t)
        self.s_tmp_t = np.minimum(self.s_tmp_t, self.S_t)
        # a priori speech absence and presence probabilities
        smin_t = np.maximum(self.s_min_t, _EPS)
        gmin_t = y2 / (bm * smin_t)
        zeta_t = self.S_t / (bm * smin_t)
        low = zeta_t < self.zeta0
        q = np.where(low & (gmin_t <= 1.0), 1.0,
                     np.where(low & (gmin_t < self.gamma1),
                              (self.gamma1 - gmin_t) / (self.gamma1 - 1.0), 0.0))
        p = _presence(q, xi, gamma)
        # recursive averaging with p-dependent smoothing, bias compensation
        a_d = self.alpha_d + (1 - self.alpha_d) * p
        self.lam_bar = np.maximum(a_d * self.lam_bar + (1 - a_d) * y2, _EPS)
        self.lam = self.beta * self.lam_bar
        # sub-window bookkeeping: every V frames store the sub-window minimum
        self.count += 1
        if self.count == self.V:
            self.count = 0
            self.buf = np.vstack([self.buf[1:], self.s_tmp[None]])
            self.buf_t = np.vstack([self.buf_t[1:], self.s_tmp_t[None]])
            self.s_min = self.buf.min(axis=0)
            self.s_min_t = self.buf_t.min(axis=0)
            self.s_tmp = self.S.copy()
            self.s_tmp_t = self.S_t.copy()
        return self.lam


class AprioriAbsence:
    """Cohen & Berdugo (2001) a priori speech absence q from recursively averaged xi."""

    def __init__(self, n_bins, beta=0.7, w_local=1, w_global=15, zeta_min_db=-10.0,
                 zeta_max_db=-5.0, zeta_p_min_db=0.0, zeta_p_max_db=10.0, q_max=0.95):
        self.beta, self.q_max = beta, q_max
        self.k_local, self.k_global = _hann_kernel(int(w_local)), _hann_kernel(int(w_global))
        self.z_lo, self.z_hi = 10 ** (zeta_min_db / 10), 10 ** (zeta_max_db / 10)
        self.zp_lo, self.zp_hi = 10 ** (zeta_p_min_db / 10), 10 ** (zeta_p_max_db / 10)
        self.zeta = np.zeros(n_bins)
        self.frame_prev = 0.0
        self.peak = self.zp_lo

    def q(self, xi_prev):
        """q for the current frame from the previous frame's xi."""
        self.zeta = self.beta * self.zeta + (1 - self.beta) * xi_prev
        p_local = _log_map(_smooth(self.zeta, self.k_local), self.z_lo, self.z_hi)
        p_global = _log_map(_smooth(self.zeta, self.k_global), self.z_lo, self.z_hi)
        zf = float(self.zeta.mean())
        if zf <= self.z_lo:
            p_frame = 0.0
        elif zf > self.frame_prev:            # onset: track the peak, assume speech
            self.peak = min(max(zf, self.zp_lo), self.zp_hi)
            p_frame = 1.0
        else:
            p_frame = float(_log_map(zf / self.peak, self.z_lo, self.z_hi))
        self.frame_prev = zf
        return np.minimum(1.0 - p_local * p_global * p_frame, self.q_max)


def validate(n_fft, hop, window, noise_est, gain_mode, noise_fraction, alpha, q_max, imcra_kw):
    check_nola(n_fft, hop, window)
    if noise_est not in ("initial", "imcra"):
        raise ValueError("noise_est must be 'initial' or 'imcra'.")
    if gain_mode not in ("lsa", "omlsa"):
        raise ValueError("gain_mode must be 'lsa' or 'omlsa'.")
    if not 0 < noise_fraction <= 1:
        raise ValueError("noise_fraction must be in (0, 1].")
    if not 0 <= q_max < 1:
        raise ValueError("q_max must be in [0, 1).")
    for name in ("alpha", "alpha_s", "alpha_d", "omlsa_beta"):
        v = alpha if name == "alpha" else imcra_kw[name]
        if not 0 <= v < 1:
            raise ValueError(f"{name} must be in [0, 1).")
    if imcra_kw["U"] < 1 or imcra_kw["V"] < 1 or imcra_kw["beta"] <= 0 or imcra_kw["b_min"] <= 0:
        raise ValueError("IMCRA needs U, V >= 1 and beta, b_min > 0.")
    if not imcra_kw["gamma1"] > 1:
        raise ValueError("gamma1 must be > 1.")


def enhance(noisy, sr=16000, n_fft=512, hop=128, window="hann", noise_est="imcra",
            gain_mode="omlsa", noise_fraction=0.20, alpha=0.92, xi_min_db=-15.0, gmin_db=-25.0,
            q_max=0.95, alpha_s=0.9, alpha_d=0.85, beta=1.47, w=1, U=8, V=15, b_min=1.66,
            gamma0=4.6, gamma1=3.0, zeta0=1.67, init_frames=6, omlsa_beta=0.7):
    x = as_waveform(noisy)
    if sr != 16000:
        raise ValueError("This method uses 16000 Hz / wideband PESQ.")
    imcra_kw = dict(alpha_s=alpha_s, alpha_d=alpha_d, beta=beta, w=w, U=U, V=V, b_min=b_min,
                    gamma0=gamma0, gamma1=gamma1, zeta0=zeta0, omlsa_beta=omlsa_beta)
    validate(n_fft, hop, window, noise_est, gain_mode, noise_fraction, alpha, q_max, imcra_kw)
    imcra_kw.pop("omlsa_beta")
    _, times, z = stft(x, sr, n_fft, hop, window)
    power = np.abs(z) ** 2
    n_bins, n_frames = z.shape
    xi_min = 10.0 ** (xi_min_db / 10.0)
    gmin = 10.0 ** (gmin_db / 20.0)

    if noise_est == "initial":
        noise0 = estimate_noise_power(z, times, len(x), sr, n_fft, noise_fraction)
        tracker = None
    else:
        # initialise from the first full-support frames (the boundary frames are half zeros)
        centers = np.rint(times * sr).astype(int)
        full = np.flatnonzero((centers >= n_fft // 2) & (centers + n_fft // 2 <= max(len(x), n_fft)))
        init = full[: max(1, int(init_frames))] if len(full) else np.arange(min(n_frames, 1))
        tracker = IMCRA(power[:, init].mean(axis=1), **imcra_kw)
        noise0 = tracker.lam
    absence = AprioriAbsence(n_bins, beta=omlsa_beta, q_max=q_max) if gain_mode == "omlsa" else None

    enhanced = np.empty_like(z)
    noise = np.maximum(noise0, _EPS)
    prev_amp2 = np.zeros(n_bins)
    xi_prev = np.zeros(n_bins)
    for t in range(n_frames):
        y2 = power[:, t]
        gamma = np.minimum(y2 / noise, 1e8)
        inst = np.maximum(gamma - 1.0, 0.0)
        if t == 0:
            xi = alpha + (1 - alpha) * inst
        else:
            xi = alpha * prev_amp2 / noise + (1 - alpha) * inst
        xi = np.maximum(xi, xi_min)
        g_h1 = np.minimum(lsa_gain(xi, gamma), 1.0)
        if absence is None:
            g = np.maximum(g_h1, gmin)
            prev_amp2 = (g * g) * y2
        else:
            p = _presence(absence.q(xi_prev), xi, gamma)
            g = g_h1 ** p * gmin ** (1.0 - p)
            prev_amp2 = (g_h1 * g_h1) * y2
        xi_prev = xi
        enhanced[:, t] = g * z[:, t]
        if tracker is not None:
            noise = np.maximum(tracker.update(y2, xi, gamma), _EPS)
    return istft(enhanced, len(x), sr, n_fft, hop, window)
