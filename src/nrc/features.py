"""Noise-aware input features for gru_mask (gru_v6b), from the dsp-v8 estimators. Pure numpy / scipy.

Per file, on dsp-v8's own STFT grid (n_fft 1024, scipy "hann" (periodic), hop 256, frame k centred at k*256 samples):
  noise  = stationary noise PSD, spp_mean (configs/dsp_p.yaml: noise_fraction 0.1, spp_iters 3, spp_xi_h1_db 10),
           BEFORE the noise bias                                          -> log noise PSD (513 bins)
  xi     = first-step decision-directed a priori SNR of the dsp-v7 path (forward DD, alpha 0.94, xi_min -25 dB,
           LSA gain, gmin -30 dB, TSNR gain feeding the recursion, noise x two-band bias 1.25/1.5), no NLM
                                                                          -> log xi (513 bins x frames)
Mapping to the model grid (n_fft 512, hop 128, torch.stft center=True: frame m centred at m*128 samples, 257 bins):
  frequency: linear interpolation of the LOG values over frequency in Hz (model bin f at f*31.25 Hz; this lands on dsp
             bin 2f, so it is exact resampling of the log spectrum, written generally as an interpolation matrix);
  time:      log noise PSD is stationary -> broadcast to every model frame; log xi is linearly interpolated over time
             in the log domain between the dsp frame centres (model frame m sits at dsp frame m/2; edges held).
Output: float32 (T, 2*257) = [log noise PSD, log xi] per model frame, T = 1 + len(x) // 128.
"""
from __future__ import annotations

import functools

import numpy as np

from .methods import mmse_lsa
from .stft import stft

SR = 16000
DSP = {"n_fft": 1024, "hop": 256, "window": "hann", "noise_fraction": 0.1, "spp_iters": 3, "spp_xi_h1_db": 10.0,
       "noise_bias_low": 1.25, "noise_bias_high": 1.5, "alpha": 0.94, "xi_min_db": -25.0, "gmin_db": -30.0}
LOG_EPS = 1e-10


@functools.lru_cache(maxsize=8)
def _freq_matrix(n_src: int, n_dst: int, sr: int = SR) -> np.ndarray:
    """(n_dst, n_src) linear interpolation over frequency in Hz (both grids span 0..sr/2)."""
    f_src = np.linspace(0.0, sr / 2, n_src)
    f_dst = np.linspace(0.0, sr / 2, n_dst)
    return _interp_matrix(f_src, f_dst)


def _interp_matrix(x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Rows = np.interp weights (edges held), so W @ v == np.interp(x_dst, x_src, v)."""
    w = np.zeros((len(x_dst), len(x_src)))
    j = np.clip(np.searchsorted(x_src, x_dst, side="right") - 1, 0, len(x_src) - 2)
    t = np.clip((x_dst - x_src[j]) / (x_src[j + 1] - x_src[j]), 0.0, 1.0)
    rows = np.arange(len(x_dst))
    w[rows, j] = 1.0 - t
    w[rows, j + 1] += t
    return w


def dsp_estimates(x: np.ndarray):
    """(times_s, noise_psd (513,), xi1 (513, frames)) on the dsp-v8 grid."""
    p = DSP
    x = np.asarray(x, dtype=np.float64)
    _, times, z = stft(x, SR, p["n_fft"], p["hop"], p["window"])
    power = np.abs(z) ** 2
    est = mmse_lsa.stationary_noise_psd(z, power, times, len(x), SR, p["n_fft"], "spp_mean", p["noise_fraction"],
                                        spp_iters=p["spp_iters"], spp_xi_h1_db=p["spp_xi_h1_db"])
    bias = mmse_lsa.two_band_bias(z.shape[0], SR, p["n_fft"], p["noise_bias_low"], p["noise_bias_high"])
    noise = np.maximum(est * bias, mmse_lsa._EPS)
    noise_tf = np.broadcast_to(noise[:, None], z.shape)
    gamma = np.minimum(power / noise_tf, 1e8)
    xi_min, gmin = 10.0 ** (p["xi_min_db"] / 10.0), 10.0 ** (p["gmin_db"] / 20.0)
    xi1, _, _ = mmse_lsa._dd_pass(gamma, noise_tf, p["alpha"], xi_min, gmin, "hrnr", mmse_lsa.lsa_gain)
    return times, est, xi1


def noise_aware_features(x, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """float32 (T, 2F) on the model grid; F = n_fft // 2 + 1, T = 1 + len(x) // hop."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    times, est, xi1 = dsp_estimates(x)
    n_bins, n_frames = n_fft // 2 + 1, 1 + len(x) // hop
    wf = _freq_matrix(est.shape[0], n_bins)
    log_noise = wf @ np.log(np.maximum(est, LOG_EPS))                      # (F,)
    log_xi = wf @ np.log(np.maximum(xi1, LOG_EPS))                         # (F, Td)
    wt = _interp_matrix(times * SR, np.arange(n_frames, dtype=float) * hop)  # (T, Td)
    log_xi_m = log_xi @ wt.T                                                # (F, T)
    out = np.concatenate([np.broadcast_to(log_noise[:, None], (n_bins, n_frames)), log_xi_m], axis=0)
    return np.ascontiguousarray(out.T, dtype=np.float32)


def batch_features(noisy: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """(B, N) -> float32 (B, T, 2F). Picklable for worker pools."""
    return np.stack([noise_aware_features(w, n_fft, hop) for w in np.asarray(noisy)])
