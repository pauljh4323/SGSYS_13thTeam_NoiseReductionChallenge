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
  hop_div                if set, hop = n_fft // hop_div (overrides hop); window may be any scipy
                         window satisfying NOLA (hann, hamming, blackman swept).
  noise_select_bands     "initial" estimator only: split 100 Hz..8 kHz into B log-spaced bands (bins
                         below 100 Hz join the lowest band) and pick the lowest-energy noise_fraction
                         of frames separately per band. B=1 = the single-band selection.
  noise_freq_smooth_bins "initial" estimator only: smooth the stationary noise PSD over +-N bins
  noise_freq_smooth_mode ("mean" or "median") before noise_bias. N=0 = off.
  hrnr_nonlinearity      HRNR rectifier: "halfwave" max(s, 0) (default) or "fullwave" |s|.
  gain_mode "beta", beta beta-order MMSE amplitude estimator (You, Koh, Rahardja 2005) for every gain
                         computation: G = sqrt(v)/gamma [Gamma(beta/2+1) 1F1(-beta/2; 1; -v)]^(1/beta).
                         beta = 1 is MMSE-STSA, beta -> 0 the LSA (used for |beta| < 1e-6). The formula
                         needs Gamma(beta/2 + 1) defined, i.e. beta > -2 (enforced here; the paper's
                         recommended range is not verified, see docs/references.md).
  pause_thr_db, pause_att_db, pause_release_frames
                         pause expander (own design): per frame, mean a posteriori SNR over 300-4000 Hz
                         (dB); below pause_thr_db the target extra attenuation is pause_att_db, else 0;
                         attack 1 frame, one-pole release exp(-1/pause_release_frames); multiplied into
                         the output gain, floored at gmin. pause_att_db 0 = off.
  noise_bias_low, noise_bias_high
                         two-band noise over-estimation (after Kamath & Loizou 2002): bias below/above
                         1 kHz with a raised-cosine cross-fade over 900-1100 Hz; None = noise_bias.
  noise_est "spp_mean"   speech-presence-weighted whole-file mean noise PSD. lambda_0 = the "initial"
                         estimate (incl. noise_select_bands / freq smoothing, before any bias); then
                         spp_iters times: gamma = |Y|^2/lambda_i, a posteriori SPP with fixed prior 0.5 and
                         fixed a priori SNR xi_H1 = 10^(spp_xi_h1_db/10) (Gerkmann & Hendriks 2012)
                         P1 = 1 / (1 + (1+xi_H1) exp(-gamma xi_H1/(1+xi_H1))), and
                         lambda_{i+1}(k) = sum_l (1-P1)|Y|^2 / sum_l (1-P1) (per bin; lambda_i kept where
                         sum_l (1-P1) < 1e-3). The whole-file iterative use is our own variant (the paper is
                         a recursive frame-wise tracker). Non-causal (offline, file-by-file).
  tv_noise, tv_alpha, tv_window_frames, tv_bias
                         upward-only time-varying correction of a stationary estimate (own design after
                         minimum statistics, Martin 2001): P = average of forward and backward one-pole
                         smoothing of |Y|^2 over time (tv_alpha), M = centred minimum of P over
                         +-tv_window_frames frames, lambda(k,l) = max(lambda_stat(k), tv_bias M(k,l)),
                         then the noise bias. Needs a stationary estimator (not "mcra"). Off by default.
  phase_consistency_iters, phase_gain_thr
                         P1 (dsp-m): K iterations X <- A exp(j angle(STFT(ISTFT(X)))) with A the enhanced
                         magnitude (Griffin & Lim 1984; consistent Wiener filtering, Le Roux & Vincent 2013);
                         bins with gain >= phase_gain_thr keep their initial (noisy) phase. K=0 = off.
  phase_harmonic, phase_beta, phase_bins, phase_reanchor, f0_min, f0_max, yin_threshold,
  harmonic_max_hz        P2 (dsp-m): harmonic phase reconstruction along time (Krawczyk & Gerkmann 2014)
                         from a YIN f0 (de Cheveigne & Kawahara 2002) of the phase-off output (incl. the
                         high-pass); see nrc/phase.py. Applied before P1 (P1 starts from its phase).
                         Both act after the magnitude gain and before the high-pass; off = bit-identical.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, minimum_filter1d, uniform_filter1d
from scipy.special import exp1, gammaln, hyp1f1

from ..io import as_waveform
from ..phase import consistency_phase, harmonic_phase, yin_f0
from ..stft import check_nola, istft, stft
from .wiener import estimate_noise_power

_EPS = 1e-18
_V_MIN, _V_MAX = 1e-8, 500.0  # exp1 finite and exp(0.5*exp1(v)) bounded in this range
_EULER = 0.5772156649015329
XI_MODES = ("dd", "tsnr", "hrnr")
HRNR_NONLINEARITIES = ("halfwave", "fullwave")
SMOOTH_MODES = ("mean", "median")
_BAND_LO_HZ = 100.0
GAIN_MODES = ("lsa", "beta")
_BETA_V_MAX = 200.0  # above: Wiener limit xi/(1+xi) (first asymptotic correction beta^2/(4v) < 1e-3)
_PAUSE_BAND_HZ = (300.0, 4000.0)
_SPLIT_HZ, _FADE_HZ = 1000.0, 200.0


def banded_initial_noise(z, times, length, sr, n_fft, noise_fraction, n_bands):
    """Lowest-energy-frame noise PSD with the frame selection done separately per band.

    Same as wiener.estimate_noise_power (complete frames only, mean of the selected frames, then a
    3-bin moving average) except that frame energy and selection are per band: B log-spaced bands
    with edges geomspace(100, sr/2, B+1); bins below 100 Hz belong to the lowest band.
    """
    if n_bands == 1:
        return estimate_noise_power(z, times, length, sr, n_fft, noise_fraction)
    power = np.abs(z) ** 2
    centers = np.rint(times * sr).astype(int)
    full = np.flatnonzero((centers >= n_fft // 2) & (centers + n_fft // 2 <= length))
    if not len(full):
        full = np.arange(power.shape[1])
    count = max(1, int(np.ceil(noise_fraction * len(full))))
    freqs = np.arange(power.shape[0]) * sr / n_fft
    edges = np.geomspace(_BAND_LO_HZ, sr / 2, n_bands + 1)
    band = np.clip(np.searchsorted(edges, freqs, side="right") - 1, 0, n_bands - 1)
    estimate = np.empty(power.shape[0])
    for b in range(n_bands):
        rows = np.flatnonzero(band == b)
        if not len(rows):
            continue
        energy = power[np.ix_(rows, full)].mean(axis=0)
        selected = full[np.argsort(energy, kind="stable")[:count]]
        estimate[rows] = power[np.ix_(rows, selected)].mean(axis=1)
    return np.maximum(uniform_filter1d(estimate, size=3, mode="nearest"), 1e-18)


def smooth_noise_freq(noise, n_bins, mode="mean"):
    """+-n_bins smoothing of a (bins,) PSD across frequency (edges: nearest)."""
    n = int(n_bins)
    if n <= 0:
        return noise
    if mode == "median":
        return median_filter(noise, size=2 * n + 1, mode="nearest")
    return uniform_filter1d(noise, size=2 * n + 1, mode="nearest")


def lsa_gain(xi, gamma):
    """G = xi/(1+xi) * exp(0.5 * E1(v)), v = xi*gamma/(1+xi)."""
    v = np.clip(xi * gamma / (1.0 + xi), _V_MIN, _V_MAX)
    return xi / (1.0 + xi) * np.exp(0.5 * exp1(v))


def beta_gain(xi, gamma, beta):
    """beta-order MMSE amplitude gain (You, Koh, Rahardja 2005); LSA for |beta| < 1e-6.

    G = sqrt(v)/gamma * [Gamma(beta/2+1) * 1F1(-beta/2; 1; -v)]^(1/beta), v = xi*gamma/(1+xi),
    evaluated in the log domain. Where v > _BETA_V_MAX or 1F1 is not finite/positive the large-v
    limit xi/(1+xi) is used. Returned gain is finite (callers clip to [gmin, 1]).
    """
    if abs(beta) < 1e-6:
        return lsa_gain(xi, gamma)
    wiener = xi / (1.0 + xi)
    v = np.clip(xi * gamma / (1.0 + xi), _V_MIN, _V_MAX)
    small = v <= _BETA_V_MAX
    g = wiener.copy()
    if np.any(small):
        vs = v[small]
        with np.errstate(all="ignore"):
            m = hyp1f1(-beta / 2.0, 1.0, -vs)
            ok = np.isfinite(m) & (m > 0)
            log_g = (0.5 * np.log(vs) - np.log(np.maximum(gamma[small], _EPS))
                     + (gammaln(beta / 2.0 + 1.0) + np.log(np.where(ok, m, 1.0))) / beta)
            gs = np.exp(np.minimum(log_g, 50.0))
        g[small] = np.where(ok & np.isfinite(gs), gs, wiener[small])
    return np.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0)


def pause_attenuation(gamma, sr, n_fft, thr_db, att_db, release_frames):
    """Per-frame extra gain factor (frames,) of the pause expander (1.0 where inactive)."""
    freqs = np.arange(gamma.shape[0]) * sr / n_fft
    band = (freqs >= _PAUSE_BAND_HZ[0]) & (freqs <= _PAUSE_BAND_HZ[1])
    snr_db = 10.0 * np.log10(np.maximum(gamma[band].mean(axis=0), _EPS))
    target = np.where(snr_db < thr_db, float(att_db), 0.0)
    a = np.exp(-1.0 / release_frames)
    att = np.empty_like(target)
    prev = 0.0
    for t, tg in enumerate(target):
        prev = tg if tg >= prev else a * prev + (1.0 - a) * tg  # attack 1 frame, one-pole release
        att[t] = prev
    return 10.0 ** (-att / 20.0)


def two_band_bias(n_bins, sr, n_fft, low, high):
    """Per-bin noise bias: low below / high above 1 kHz, raised-cosine cross-fade over 900-1100 Hz."""
    freqs = np.arange(n_bins) * sr / n_fft
    x = np.clip((freqs - (_SPLIT_HZ - _FADE_HZ / 2)) / _FADE_HZ, 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * x)
    return low * (1.0 - w) + high * w


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


NOISE_ESTS = ("initial", "mcra", "quantile", "spp_mean")
STATIONARY_ESTS = ("initial", "quantile", "spp_mean")
_SPP_DEN_MIN = 1e-3


def spp_mean_noise(power, lambda0, iters=2, xi_h1_db=15.0):
    """Speech-presence-weighted whole-file mean noise PSD (bins,), started from lambda0 (bins,)."""
    xi = 10.0 ** (xi_h1_db / 10.0)
    lam = np.maximum(np.asarray(lambda0, dtype=float), _EPS)
    for _ in range(int(iters)):
        gamma = power / lam[:, None]
        p1 = 1.0 / (1.0 + (1.0 + xi) * np.exp(-gamma * xi / (1.0 + xi)))
        w = 1.0 - p1
        den = w.sum(axis=1)
        num = (w * power).sum(axis=1)
        ok = den >= _SPP_DEN_MIN
        lam = np.maximum(np.where(ok, num / np.where(ok, den, 1.0), lam), _EPS)
    return lam


def tv_noise_floor(power, stat, alpha=0.8, window_frames=16, bias=1.0):
    """Upward-only time-varying noise PSD (bins, frames): max(stat, bias * local min of smoothed |Y|^2)."""
    fwd = signal.lfilter([1.0 - alpha], [1.0, -alpha], power, axis=1,
                         zi=(alpha * power[:, :1]))[0]
    bwd = signal.lfilter([1.0 - alpha], [1.0, -alpha], power[:, ::-1], axis=1,
                         zi=(alpha * power[:, -1:]))[0][:, ::-1]
    p = 0.5 * (fwd + bwd)
    m = minimum_filter1d(p, size=2 * int(window_frames) + 1, axis=1, mode="nearest")
    return np.maximum(stat[:, None], bias * m)


def stationary_noise_psd(z, power, times, length, sr, n_fft, noise_est="initial",
                         noise_fraction=0.20, q=0.2, bias=1.0, smooth_frames=0,
                         noise_select_bands=1, noise_freq_smooth_bins=0,
                         noise_freq_smooth_mode="mean", spp_iters=2, spp_xi_h1_db=15.0):
    """Stationary noise PSD (bins,) before noise_bias / two-band bias (quantile includes its own bias)."""
    if noise_est == "quantile":
        return quantile_noise(power, q, bias, smooth_frames)
    est = banded_initial_noise(z, times, length, sr, n_fft, noise_fraction, int(noise_select_bands))
    est = smooth_noise_freq(est, noise_freq_smooth_bins, noise_freq_smooth_mode)
    if noise_est == "spp_mean":
        est = spp_mean_noise(power, est, spp_iters, spp_xi_h1_db)
    return est


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
             xi_mode="dd", hrnr_rho=0.5, highpass_hz=0.0, sr=16000, noise_select_bands=1,
             noise_freq_smooth_bins=0, noise_freq_smooth_mode="mean",
             hrnr_nonlinearity="halfwave", gain_mode="lsa", beta=0.0, pause_thr_db=3.0,
             pause_att_db=0.0, pause_release_frames=4, noise_bias_low=None, noise_bias_high=None,
             spp_iters=2, spp_xi_h1_db=15.0, tv_noise=False, tv_alpha=0.8, tv_window_frames=16,
             tv_bias=1.0, phase_consistency_iters=0, phase_gain_thr=1.0, phase_harmonic=False,
             phase_beta=1.0, phase_bins=1, phase_reanchor=True, f0_min=70.0, f0_max=400.0,
             yin_threshold=0.15, harmonic_max_hz=4000.0):
    check_nola(n_fft, hop, window)
    if int(phase_consistency_iters) != phase_consistency_iters or phase_consistency_iters < 0:
        raise ValueError("phase_consistency_iters must be an integer >= 0.")
    if not phase_gain_thr > 0:
        raise ValueError("phase_gain_thr must be > 0.")
    if phase_harmonic:
        if not 0 <= phase_beta <= 1:
            raise ValueError("phase_beta must be in [0, 1].")
        if int(phase_bins) != phase_bins or phase_bins < 0:
            raise ValueError("phase_bins must be an integer >= 0.")
        if not 0 < f0_min < f0_max < sr / 4:
            raise ValueError("need 0 < f0_min < f0_max < sr/4.")
        if not 0 < yin_threshold < 1:
            raise ValueError("yin_threshold must be in (0, 1).")
        if not 0 < harmonic_max_hz <= sr / 2:
            raise ValueError("harmonic_max_hz must be in (0, sr/2].")
    if int(spp_iters) != spp_iters or spp_iters < 1:
        raise ValueError("spp_iters must be an integer >= 1.")
    if not np.isfinite(spp_xi_h1_db):
        raise ValueError("spp_xi_h1_db must be finite.")
    if tv_noise:
        if noise_est not in STATIONARY_ESTS:
            raise ValueError(f"tv_noise needs a stationary noise_est {STATIONARY_ESTS}.")
        if not 0 <= tv_alpha < 1:
            raise ValueError("tv_alpha must be in [0, 1).")
        if int(tv_window_frames) != tv_window_frames or tv_window_frames < 0:
            raise ValueError("tv_window_frames must be an integer >= 0.")
        if tv_bias <= 0:
            raise ValueError("tv_bias must be > 0.")
    if gain_mode not in GAIN_MODES:
        raise ValueError(f"gain_mode must be one of {GAIN_MODES}.")
    if gain_mode == "beta" and not beta > -2:
        raise ValueError("beta must be > -2 (Gamma(beta/2 + 1) undefined otherwise).")
    if pause_att_db < 0 or pause_release_frames <= 0:
        raise ValueError("pause_att_db must be >= 0 and pause_release_frames > 0.")
    for name, v in (("noise_bias_low", noise_bias_low), ("noise_bias_high", noise_bias_high)):
        if v is not None and v <= 0:
            raise ValueError(f"{name} must be > 0.")
    if int(noise_select_bands) != noise_select_bands or noise_select_bands < 1:
        raise ValueError("noise_select_bands must be an integer >= 1.")
    if int(noise_freq_smooth_bins) != noise_freq_smooth_bins or noise_freq_smooth_bins < 0:
        raise ValueError("noise_freq_smooth_bins must be an integer >= 0.")
    if noise_freq_smooth_mode not in SMOOTH_MODES:
        raise ValueError(f"noise_freq_smooth_mode must be one of {SMOOTH_MODES}.")
    if hrnr_nonlinearity not in HRNR_NONLINEARITIES:
        raise ValueError(f"hrnr_nonlinearity must be one of {HRNR_NONLINEARITIES}.")
    if (noise_est not in ("initial", "spp_mean")
            and (noise_select_bands != 1 or noise_freq_smooth_bins)):
        raise ValueError("noise_select_bands / noise_freq_smooth_bins need noise_est "
                         "'initial' or 'spp_mean'.")
    if noise_bias <= 0:
        raise ValueError("noise_bias must be > 0.")
    if xi_mode not in XI_MODES:
        raise ValueError(f"xi_mode must be one of {XI_MODES}.")
    if not 0 <= hrnr_rho <= 1:
        raise ValueError("hrnr_rho must be in [0, 1].")
    if not (highpass_hz == 0 or 0 < highpass_hz < sr / 2):
        raise ValueError("highpass_hz must be 0 (off) or in (0, sr/2).")
    if noise_est not in NOISE_ESTS:
        raise ValueError(f"noise_est must be one of {NOISE_ESTS}.")
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
            xi_cepstral_smoothing=False, hrnr_rho=0.5, highpass_hz=0.0, hop_div=None,
            noise_select_bands=1, noise_freq_smooth_bins=0, noise_freq_smooth_mode="mean",
            hrnr_nonlinearity="halfwave", gain_mode="lsa", beta=0.0, pause_thr_db=3.0,
            pause_att_db=0.0, pause_release_frames=4, noise_bias_low=None, noise_bias_high=None,
            spp_iters=2, spp_xi_h1_db=15.0, tv_noise=False, tv_alpha=0.8, tv_window_frames=16,
            tv_bias=1.0, phase_consistency_iters=0, phase_gain_thr=1.0, phase_harmonic=False,
            phase_beta=1.0, phase_bins=1, phase_reanchor=True, f0_min=70.0, f0_max=400.0,
            yin_threshold=0.15, harmonic_max_hz=4000.0):
    x = as_waveform(noisy)
    if sr != 16000:
        raise ValueError("This method uses 16000 Hz / wideband PESQ.")
    if hop_div is not None:
        if int(hop_div) != hop_div or hop_div < 1 or n_fft % int(hop_div):
            raise ValueError("hop_div must be a positive integer dividing n_fft.")
        hop = n_fft // int(hop_div)
    validate(n_fft, hop, window, noise_est, noise_fraction, alpha, alpha_d, alpha_p, alpha_s,
             mcra_window, delta, q, bias, smooth_frames, noise_bias, xi_mode, hrnr_rho,
             highpass_hz, sr, noise_select_bands, noise_freq_smooth_bins, noise_freq_smooth_mode,
             hrnr_nonlinearity, gain_mode, beta, pause_thr_db, pause_att_db, pause_release_frames,
             noise_bias_low, noise_bias_high, spp_iters, spp_xi_h1_db, tv_noise, tv_alpha,
             tv_window_frames, tv_bias, phase_consistency_iters, phase_gain_thr, phase_harmonic,
             phase_beta, phase_bins, phase_reanchor, f0_min, f0_max, yin_threshold, harmonic_max_hz)
    _, times, z = stft(x, sr, n_fft, hop, window)
    power = np.abs(z) ** 2
    if noise_est in STATIONARY_ESTS:
        est = stationary_noise_psd(z, power, times, len(x), sr, n_fft, noise_est, noise_fraction, q,
                                   bias, smooth_frames, noise_select_bands, noise_freq_smooth_bins,
                                   noise_freq_smooth_mode, spp_iters, spp_xi_h1_db)
        if tv_noise:
            noise = tv_noise_floor(power, est, tv_alpha, tv_window_frames, tv_bias)
        else:
            noise = np.repeat(est[:, None], z.shape[1], axis=1)
    else:
        noise = mcra_noise(power, alpha_d, alpha_p, alpha_s, int(mcra_window), delta)
    if noise_bias_low is not None or noise_bias_high is not None:
        lo = noise_bias if noise_bias_low is None else noise_bias_low
        hi = noise_bias if noise_bias_high is None else noise_bias_high
        noise = noise * two_band_bias(noise.shape[0], sr, n_fft, lo, hi)[:, None]
    elif noise_bias != 1.0:
        noise = noise * noise_bias
    noise = np.maximum(noise, _EPS)
    xi_min = 10.0 ** (xi_min_db / 10.0)
    gmin = 10.0 ** (gmin_db / 20.0)
    gamma = np.minimum(power / noise, 1e8)
    if gain_mode == "lsa":
        gain_fn = lsa_gain
    else:
        def gain_fn(xi_, g_):
            return beta_gain(xi_, g_, float(beta))
    pause = (pause_attenuation(gamma, sr, n_fft, pause_thr_db, pause_att_db, pause_release_frames)
             if pause_att_db > 0 else None)
    if xi_mode != "dd" or xi_cepstral_smoothing:
        s_hat, g = _enhance_two_step(z, gamma, noise, alpha, xi_min, gmin, xi_mode,
                              bool(xi_cepstral_smoothing), hrnr_rho, len(x), sr, n_fft, hop, window,
                              hrnr_nonlinearity, gain_fn, pause)
    else:
        s_hat, g = _enhance_dd(z, gamma, noise, alpha, xi_min, gmin, len(x), sr, n_fft, hop, window,
                        gain_fn, pause)
    y = _highpass(istft(s_hat, len(x), sr, n_fft, hop, window), highpass_hz, sr)
    if not phase_consistency_iters and not phase_harmonic:
        return y
    phi_noisy = np.angle(z)
    phase = phi_noisy
    if phase_harmonic:
        f0 = yin_f0(y, z.shape[1], sr, hop, f0_min, f0_max, yin_threshold)
        phase = harmonic_phase(phi_noisy, g, f0, sr, n_fft, hop, phase_gain_thr, phase_beta,
                               int(phase_bins), harmonic_max_hz, bool(phase_reanchor), f0_min)
    amp = np.abs(s_hat)
    if phase_consistency_iters:
        phase = consistency_phase(amp, phase, g >= phase_gain_thr, int(phase_consistency_iters),
                                  sr, n_fft, hop, window)
    return _highpass(istft(amp * np.exp(1j * phase), len(x), sr, n_fft, hop, window),
                     highpass_hz, sr)


def _highpass(y, highpass_hz, sr):
    if not highpass_hz:
        return y
    sos = signal.butter(4, highpass_hz, btype="highpass", fs=sr, output="sos")
    if len(y) <= 3 * (2 * len(sos) + 1):  # too short for sosfiltfilt's default padding
        return y
    return signal.sosfiltfilt(sos, y)


def _enhance_dd(z, gamma, noise, alpha, xi_min, gmin, length, sr, n_fft, hop, window,
                gain_fn=lsa_gain, pause=None):
    """Plain decision-directed LSA (the dsp-v1 path): returns (enhanced STFT, gain)."""
    enhanced = np.empty_like(z)
    g_all = np.empty(z.shape)
    prev_amp2 = np.zeros(z.shape[0])
    for t in range(z.shape[1]):
        inst = np.maximum(gamma[:, t] - 1.0, 0.0)
        if t == 0:
            xi = alpha + (1 - alpha) * inst  # standard DD initialisation
        else:
            xi = alpha * prev_amp2 / noise[:, t] + (1 - alpha) * inst
        xi = np.maximum(xi, xi_min)
        g = np.clip(gain_fn(xi, gamma[:, t]), gmin, 1.0)
        g_all[:, t] = g
        enhanced[:, t] = g * z[:, t]
        prev_amp2 = np.abs(enhanced[:, t]) ** 2
    if pause is not None:  # output gain only; the DD recursion above is unchanged
        g_all = np.maximum(g_all * pause[None, :], gmin)
        enhanced = g_all * z
    return enhanced, g_all


def _enhance_two_step(z, gamma, noise, alpha, xi_min, gmin, xi_mode, cepstral, rho, length, sr,
                      n_fft, hop, window, nonlinearity="halfwave", gain_fn=lsa_gain, pause=None):
    """First-step xi (DD or cepstral), then optional TSNR / HRNR second step: (STFT, gain)."""
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
        g = np.clip(gain_fn(xi1, gm), gmin, 1.0)
        if xi_mode != "dd":  # TSNR second step: xi2 = |G1 Y|^2 / lambda_d = G1^2 gamma
            xi2 = np.maximum(g * g * gm, xi_min)
            g = np.clip(gain_fn(xi2, gm), gmin, 1.0)
        g_final[:, t] = g
        prev_amp2 = g * g * gm * noise[:, t]  # |S_hat|^2 of the final estimate (DD recursion)
    if xi_mode == "hrnr":
        s_tsnr = istft(g_final * z, length, sr, n_fft, hop, window)
        rect = np.abs(s_tsnr) if nonlinearity == "fullwave" else np.maximum(s_tsnr, 0.0)
        _, _, z_h = stft(rect, sr, n_fft, hop, window)
        z_h = z_h[:, :z.shape[1]]
        xi_h = (rho * g_final ** 2 * np.abs(z) ** 2 + (1 - rho) * np.abs(z_h) ** 2) / noise
        g_final = np.clip(gain_fn(np.maximum(xi_h, xi_min), gamma), gmin, 1.0)
    if pause is not None:
        g_final = np.maximum(g_final * pause[None, :], gmin)
    return g_final * z, g_final
