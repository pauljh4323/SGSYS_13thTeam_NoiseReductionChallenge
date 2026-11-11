"""Classical phase enhancement helpers for mmse_lsa (dsp-m): YIN f0, harmonic phase, STFT consistency.

Conventions (scipy.signal.stft as wrapped in nrc.stft, boundary="zeros", periodic Hann or any window
symmetric about n_fft/2): frame l covers samples l*hop - n_fft/2 ... l*hop + n_fft/2 - 1 of the signal,
so its centre is sample l*hop, and the FFT time origin is the frame START. For a sinusoid of
frequency f whose phase at the frame centre is phi_c, every bin k inside the window main lobe has
STFT phase phi_c - (2*pi*k/n_fft)*(n_fft/2) = phi_c - pi*k (the window DTFT is real times
exp(-j*theta*n_fft/2)). This is the "analysis-window phase offset" used below.

  yin_f0            de Cheveigne & Kawahara (2002) style f0 per STFT frame: difference function,
                    cumulative-mean normalisation, absolute threshold, parabolic interpolation.
                    d(tau) compares samples j and j+tau of the window, so its effective time is
                    start + win/2 + tau/2, not the frame centre; each voiced run is re-sampled to the
                    frame centres by linear interpolation in time (removes a glide-dependent bias).
                    Frames whose normalised minimum stays above the threshold are unvoiced (0).
  harmonic_phase    Krawczyk & Gerkmann (2014) style phase reconstruction along time for voiced
                    frames: phi_c,h(l) = phi_c,h(l-1) + 2*pi*f_h*hop/fs with f_h the mean of the
                    harmonic frequency at l-1 and l (trapezoid), initialised (and, with reanchor,
                    reset where the bin gain >= thr) from the noisy phase phi_noisy(k_h) + pi*k_h.
                    A relative f0 jump > _F0_JUMP between frames is treated as a new onset (own choice).
  consistency_phase Griffin & Lim (1984) style iterations X <- A exp(j angle(STFT(ISTFT(X)))) with
                    the noisy phase kept in reliable bins (gain >= thr), cf. Le Roux & Vincent (2013).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

_F0_JUMP = 0.2
_SILENT = 1e-10


def yin_f0(y, n_frames, sr=16000, hop=256, f0_min=70.0, f0_max=400.0, threshold=0.15,
           win=512):
    """f0 in Hz per STFT frame (n_frames,), 0 = unvoiced; window centred on sample l*hop."""
    y = np.asarray(y, dtype=float)
    tau_min = max(2, int(np.floor(sr / f0_max)))
    tau_max = int(np.ceil(sr / f0_min))
    seg = win + tau_max
    half = seg // 2
    padded = np.pad(y, (half, (n_frames - 1) * hop + seg))
    idx = np.arange(n_frames)[:, None] * hop + np.arange(seg)[None, :]
    frames = padded[idx]  # frame l starts at sample l*hop - half
    nfft = 1 << int(np.ceil(np.log2(seg + win)))
    a = np.fft.rfft(frames[:, :win], nfft, axis=1)
    b = np.fft.rfft(frames, nfft, axis=1)
    cross = np.fft.irfft(np.conj(a) * b, nfft, axis=1)[:, :tau_max + 1]  # sum_j x_j x_{j+tau}
    sq = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 1)
    e0 = sq[:, win][:, None]
    e_tau = sq[:, taus + win] - sq[:, taus]
    d = np.maximum(e0 + e_tau - 2.0 * cross, 0.0)
    cum = np.cumsum(d[:, 1:], axis=1)
    dn = np.ones_like(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        dn[:, 1:] = np.where(cum > 0, d[:, 1:] * taus[1:] / cum, 1.0)
    f0 = np.zeros(n_frames)
    for l in range(n_frames):
        if e0[l, 0] < _SILENT:
            continue
        row = dn[l]
        below = np.flatnonzero(row[tau_min:tau_max] < threshold)
        if not len(below):
            continue
        t = tau_min + below[0]
        while t + 1 < tau_max and row[t + 1] < row[t]:
            t += 1
        a1, a2, a3 = row[t - 1], row[t], row[t + 1]
        den = a1 - 2.0 * a2 + a3
        shift = 0.5 * (a1 - a3) / den if den > 0 else 0.0
        f0[l] = sr / (t + float(np.clip(shift, -0.5, 0.5)))
    centre = np.arange(n_frames) * hop
    t_eff = centre - half + win / 2 + np.where(f0 > 0, sr / np.maximum(f0, 1e-9), 0.0) / 2
    out = f0.copy()
    voiced = f0 > 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a > 1:
            out[a:b] = np.interp(centre[a:b], t_eff[a:b], f0[a:b])
    return out


def harmonic_phase(phi_noisy, gain, f0, sr=16000, n_fft=1024, hop=256, thr=0.6, beta=1.0,
                   bins=1, max_hz=4000.0, reanchor=True, f0_min=70.0):
    """Phase (bins, frames) with harmonic bins of voiced frames reconstructed and blended.

    Only bins with gain < thr are changed: phi = angle(beta e^{j phi_rec} + (1-beta) e^{j phi_noisy}).
    """
    n_bins, n_frames = phi_noisy.shape
    h_max = int(np.ceil(max_hz / f0_min)) + 1
    state = np.zeros(h_max + 1)
    alive = np.zeros(h_max + 1, dtype=bool)
    rec = phi_noisy.copy()
    mask = np.zeros(phi_noisy.shape, dtype=bool)
    prev_f = 0.0
    offs = np.arange(-int(bins), int(bins) + 1)
    for l in range(n_frames):
        f = float(f0[l])
        if f <= 0:
            alive[:] = False
            prev_f = 0.0
            continue
        if prev_f > 0 and abs(f / prev_f - 1.0) > _F0_JUMP:
            alive[:] = False
            prev_f = 0.0
        h = np.arange(1, h_max + 1)
        h = h[h * f < max_hz]
        kh = np.rint(h * f * n_fft / sr).astype(int)
        ok = kh < n_bins
        h, kh = h[ok], kh[ok]
        anchor = phi_noisy[kh, l] + np.pi * kh
        if prev_f > 0:
            prop = state[h] + 2.0 * np.pi * h * 0.5 * (f + prev_f) * hop / sr
            use_prop = alive[h]
            if reanchor:
                use_prop &= gain[kh, l] < thr
            phi_c = np.where(use_prop, prop, anchor)
        else:
            phi_c = anchor
        alive[:] = False
        state[h] = phi_c
        alive[h] = True
        prev_f = f
        k = kh[:, None] + offs[None, :]
        valid = (k >= 0) & (k < n_bins)
        vals = phi_c[:, None] - np.pi * k
        rec[k[valid], l] = vals[valid]
        mask[k[valid], l] = True
    change = mask & (gain < thr)
    out = phi_noisy.copy()
    if beta >= 1.0:
        out[change] = rec[change]
    else:
        mix = beta * np.exp(1j * rec[change]) + (1.0 - beta) * np.exp(1j * phi_noisy[change])
        out[change] = np.angle(mix)
    return np.angle(np.exp(1j * out))


def consistency_phase(amp, phase0, keep, iters, sr=16000, n_fft=1024, hop=256, window="hann"):
    """iters rounds of phase <- angle(STFT(ISTFT(amp e^{j phase}))); phase0 kept where keep."""
    phase = phase0.copy()
    n_frames = amp.shape[1]
    kw = dict(fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft)
    for _ in range(int(iters)):
        _, y = signal.istft(amp * np.exp(1j * phase), input_onesided=True, boundary=True, **kw)
        _, _, z = signal.stft(y, boundary="zeros", padded=True, **kw)
        if z.shape[1] < n_frames:
            z = np.pad(z, ((0, 0), (0, n_frames - z.shape[1])))
        phase = np.where(keep, phase0, np.angle(z[:, :n_frames]))
    return phase
