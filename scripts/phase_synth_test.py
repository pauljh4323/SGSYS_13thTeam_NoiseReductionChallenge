"""Synthetic correctness test for the dsp-m harmonic phase reconstruction (nrc/phase.py).

Signal (16 kHz, 2.0 s): two voiced segments 0.20-0.95 s and 1.20-1.95 s (silence elsewhere, 10 ms raised-cosine
ramps), 10 harmonics with amplitude 1/h and random start phases (seed 0), f0 gliding linearly 150 -> 165 Hz over
the file. STFT = dsp_k (Hann 1024 / hop 256, scipy convention).

PASS/FAIL (clean, phase_reanchor off): f0 from yin_f0 on the clean signal, initial phase = the clean STFT phase
(stands in for the noisy phase at onset), gain 0 everywhere (every harmonic bin reconstructed, beta 1,
phase_bins 1). Error = mean |wrap(phi_rec - phi_true)| over bins k_h +-1 (true f0) of frames whose window lies
fully inside a voiced segment, excluding frames where the track (re)starts. Must be < 0.3 rad.
Diagnostics only: (a) the same with the true f0; (b) white noise at 5 dB SNR, f0 from the noisy signal, gain
and noisy phase from the dsp_k pipeline (phase_gain_thr 0.6), reanchor on vs off, vs the noisy-phase error.
Writes logs/phase_synth_test.txt. Exit code 1 on FAIL.
"""
from __future__ import annotations

import sys

import numpy as np
import yaml

from _common import REPO_ROOT

from nrc.methods import mmse_lsa as ml
from nrc.phase import harmonic_phase, yin_f0
from nrc.stft import stft

SR, N_FFT, HOP = 16000, 1024, 256
SEGS = ((0.20, 0.95), (1.20, 1.95))
DUR, H, THR = 2.0, 10, 0.3


def synth(seed=0):
    t = np.arange(int(DUR * SR)) / SR
    f0 = 150.0 + 15.0 * t / DUR
    ph = 2 * np.pi * np.cumsum(f0) / SR
    rng = np.random.default_rng(seed)
    env = np.zeros_like(t)
    for a, b in SEGS:
        r = 0.010
        env += np.clip(np.minimum((t - a) / r, (b - t) / r), 0, 1)
    env = 0.5 - 0.5 * np.cos(np.pi * env)
    x = sum(np.cos(h * ph + rng.uniform(-np.pi, np.pi)) / h for h in range(1, H + 1))
    return 0.1 * env * x, f0


def eval_frames(n_frames):
    c = np.arange(n_frames) * HOP / SR
    w = N_FFT / 2 / SR
    return np.array([any(a + w <= ci <= b - w for a, b in SEGS) for ci in c])


def error(phi, phi_true, f0_frames, frames):
    errs = []
    for l in np.flatnonzero(frames):
        kh = np.rint(np.arange(1, H + 1) * f0_frames[l] * N_FFT / SR).astype(int)
        k = np.unique((kh[:, None] + np.arange(-1, 2)[None, :]).ravel())
        errs.append(np.abs(np.angle(np.exp(1j * (phi[k, l] - phi_true[k, l])))))
    return float(np.mean(np.concatenate(errs))) if errs else np.nan


def low_gain_share(g, f0_frames, frames):
    vals = []
    for l in np.flatnonzero(frames):
        kh = np.rint(np.arange(1, H + 1) * f0_frames[l] * N_FFT / SR).astype(int)
        vals.append(g[kh, l] < 0.6)
    return float(np.mean(np.concatenate(vals)))


def restarts(f0):
    """Frames where harmonic_phase (re)starts its track (anchor = no propagation)."""
    out = np.zeros(len(f0), dtype=bool)
    prev = 0.0
    for l, f in enumerate(f0):
        out[l] = f > 0 and (prev <= 0 or abs(f / prev - 1) > 0.2)
        prev = f
    return out


def run():
    x, f0_t = synth()
    _, _, z = stft(x, SR, N_FFT, HOP)
    phi_true = np.angle(z)
    n = z.shape[1]
    f0_true = f0_t[np.minimum(np.arange(n) * HOP, len(x) - 1)]
    voiced_true = np.zeros(n, dtype=bool)
    for a, b in SEGS:
        c = np.arange(n) * HOP / SR
        voiced_true |= (c >= a) & (c <= b)
    f0_true = np.where(voiced_true, f0_true, 0.0)
    frames = eval_frames(n)
    zero_gain = np.zeros(z.shape)
    lines = []

    f0_yin = yin_f0(x, n, SR, HOP)
    yin_err = np.abs(f0_yin[frames] - f0_true[frames])
    rec = harmonic_phase(phi_true, zero_gain, f0_yin, SR, N_FFT, HOP, thr=1.0, beta=1.0, bins=1,
                         reanchor=False)
    ev = frames & (f0_yin > 0) & ~restarts(f0_yin)
    e_clean = error(rec, phi_true, f0_true, ev)
    lines.append(f"YIN on clean: voiced eval frames detected {np.mean(f0_yin[frames] > 0):.3f}, "
                 f"|f0 err| mean {yin_err.mean():.4f} Hz max {yin_err.max():.4f} Hz")
    lines.append(f"PASS/FAIL clean, YIN f0, reanchor off: mean |phase err| = {e_clean:.4f} rad "
                 f"over {ev.sum()} frames (threshold {THR})")
    # error vs frames since onset (clean, YIN)
    since = np.zeros(n, dtype=int)
    for l in range(1, n):
        since[l] = since[l - 1] + 1 if (f0_yin[l] > 0 and not restarts(f0_yin)[l]) else 0
    for lo, hi in ((1, 5), (6, 15), (16, 30), (31, 99)):
        sel = ev & (since >= lo) & (since <= hi)
        if sel.any():
            lines.append(f"  frames since onset {lo}-{hi}: {error(rec, phi_true, f0_true, sel):.4f} rad "
                         f"({sel.sum()} frames)")

    rec_t = harmonic_phase(phi_true, zero_gain, f0_true, SR, N_FFT, HOP, thr=1.0, beta=1.0, bins=1,
                           reanchor=False)
    ev_t = frames & ~restarts(f0_true)
    lines.append(f"diag (a) clean, true f0, reanchor off: {error(rec_t, phi_true, f0_true, ev_t):.4f} rad")

    # (b) white noise at 5 dB, dsp_k pipeline gain / noisy phase, f0 from the noisy signal
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(x))
    noise *= np.sqrt(np.mean(x ** 2) / np.mean(noise ** 2) / 10 ** 0.5)
    y = x + noise
    params = yaml.safe_load(open(REPO_ROOT / "configs" / "dsp_k.yaml", encoding="utf-8"))["params"]
    captured = {}
    orig = ml.harmonic_phase

    def grab(phi_noisy, g, f0, *a, **k):
        captured.update(phi=phi_noisy, g=g, f0=f0)
        return phi_noisy
    ml.harmonic_phase = grab
    try:
        ml.enhance(y, SR, **params, phase_harmonic=True)
    finally:
        ml.harmonic_phase = orig
    phi_n, g = captured["phi"], captured["g"]
    lines.append(f"diag (b) 5 dB white noise: noisy-phase err {error(phi_n, phi_true, f0_true, frames):.4f} rad; "
                 f"bins with gain < 0.6 in eval harmonic bins: {low_gain_share(g, f0_true, frames):.3f}")
    for label, f0_n in (("f0 from noisy signal", yin_f0(y, n, SR, HOP)),
                        ("f0 from dsp_k output (pipeline)", captured["f0"])):
        vf = frames & (f0_n > 0)
        ferr = np.abs(f0_n - f0_true)[vf].mean() if vf.any() else np.nan
        lines.append(f"  {label}: voiced eval frames {np.mean(f0_n[frames] > 0):.3f}, "
                     f"|f0 err| mean {ferr:.4f} Hz")
        for ra in (True, False):
            r = harmonic_phase(phi_n, g, f0_n, SR, N_FFT, HOP, thr=0.6, beta=1.0, bins=1, reanchor=ra)
            lines.append(f"    reanchor {'on ' if ra else 'off'} (thr 0.6): "
                         f"{error(r, phi_true, f0_true, frames):.4f} rad (all eval frames)")
    return e_clean, lines


def main() -> int:
    e, lines = run()
    ok = e < THR
    lines.append("RESULT: " + ("PASS" if ok else "FAIL"))
    text = "\n".join(lines)
    print(text)
    (REPO_ROOT / "logs").mkdir(exist_ok=True)
    (REPO_ROOT / "logs" / "phase_synth_test.txt").write_text(text + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
