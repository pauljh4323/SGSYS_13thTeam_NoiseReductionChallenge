"""torch-pesq (audiolabs/torch-pesq 0.1.2, MIT) without torchaudio: training / analysis only, never imported by main.py.

torch-pesq imports three things from torchaudio, which has no release for torch >= 2.12 (last release 2.11.1):
  - transforms.Resample(sr, 16000): identity at 16 kHz (the only rate used here; other rates raise),
  - transforms.Spectrogram(..., power=2, center=False): |torch.stft|^2 with the same window / framing,
  - functional.lfilter(x, a, b, clamp=False): a fixed IIR filter with zero initial state. On an N-sample input
    this is exactly the causal convolution with the first N samples of the filter's impulse response
    (y[n] = sum_{k<=n} h[k] x[n-k]), so it is computed as an FFT convolution with h[0:N] (h in float64 via
    scipy.signal.lfilter, cached per length). Exact up to float rounding and differentiable.
If torchaudio is missing, a stub module is registered so that `import torch_pesq` works. The three names are then
always replaced in torch_pesq.loss by the versions here, so results are identical whether or not torchaudio exists.
"""
from __future__ import annotations

import sys
import types
from functools import lru_cache

import numpy as np
import torch


class Resample(torch.nn.Module):
    def __init__(self, orig_freq: int = 16000, new_freq: int = 16000, **_):
        super().__init__()
        if int(orig_freq) != int(new_freq):
            raise ValueError(f"nrc.tpesq supports 16 kHz only (got {orig_freq} -> {new_freq})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Spectrogram(torch.nn.Module):
    """torchaudio.transforms.Spectrogram subset used by torch-pesq (pad 0, onesided, power 2 or None)."""

    def __init__(self, n_fft: int = 400, win_length: int | None = None, hop_length: int | None = None,
                 window_fn=torch.hann_window, power: float | None = 2.0, normalized: bool = False,
                 center: bool = True, pad_mode: str = "reflect", **_):
        super().__init__()
        self.n_fft, self.win_length = n_fft, win_length or n_fft
        self.hop_length = hop_length or self.win_length // 2
        self.register_buffer("window", window_fn(self.win_length), persistent=False)
        self.power, self.normalized, self.center, self.pad_mode = power, normalized, center, pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        spec = torch.stft(x.reshape(-1, shape[-1]), self.n_fft, self.hop_length, self.win_length,
                          window=self.window.to(x.device, x.dtype), center=self.center, pad_mode=self.pad_mode,
                          normalized=self.normalized, onesided=True, return_complex=True)
        spec = spec.reshape(shape[:-1] + spec.shape[-2:])
        if self.power is None:
            return spec
        return spec.abs().pow(self.power)


@lru_cache(maxsize=32)
def _impulse(n: int, a: tuple, b: tuple) -> np.ndarray:
    from scipy.signal import lfilter as sp_lfilter
    x = np.zeros(n)
    x[0] = 1.0
    return sp_lfilter(np.asarray(b, np.float64), np.asarray(a, np.float64), x)


def lfilter(waveform: torch.Tensor, a_coeffs: torch.Tensor, b_coeffs: torch.Tensor, clamp: bool = True,
            batching: bool = True) -> torch.Tensor:
    """Zero-state IIR filter along the last axis as an exact FFT convolution (see module docstring)."""
    n = waveform.shape[-1]
    a = tuple(float(v) for v in a_coeffs.detach().cpu().double().flatten())
    b = tuple(float(v) for v in b_coeffs.detach().cpu().double().flatten())
    h = torch.as_tensor(_impulse(n, a, b), dtype=waveform.dtype, device=waveform.device)
    nfft = 1 << (2 * n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(waveform, nfft) * torch.fft.rfft(h, nfft), nfft)[..., :n]
    return y.clamp(-1.0, 1.0) if clamp else y


def _ensure_importable() -> None:
    try:
        import torchaudio  # noqa: F401
        return
    except Exception:  # missing, or a build that does not match torch
        pass
    ta = types.ModuleType("torchaudio")
    fn, tr = types.ModuleType("torchaudio.functional"), types.ModuleType("torchaudio.transforms")
    fn.lfilter = lfilter
    tr.Spectrogram, tr.Resample = Spectrogram, Resample
    ta.functional, ta.transforms = fn, tr
    ta.__nrc_stub__ = True
    sys.modules.update({"torchaudio": ta, "torchaudio.functional": fn, "torchaudio.transforms": tr})


def pesq_loss(factor: float = 1.0, sample_rate: int = 16000) -> torch.nn.Module:
    """torch_pesq.PesqLoss(factor, sample_rate) with this module's lfilter / Spectrogram / Resample.
    .mos(ref, deg) -> (B,) MOS estimate; forward(ref, deg) -> (B,) factor * (0.1 d_symm + 0.0309 d_asymm)."""
    _ensure_importable()
    import torch_pesq.loss as tl
    tl.lfilter, tl.Spectrogram, tl.Resample = lfilter, Spectrogram, Resample
    return tl.PesqLoss(factor, sample_rate=sample_rate)
