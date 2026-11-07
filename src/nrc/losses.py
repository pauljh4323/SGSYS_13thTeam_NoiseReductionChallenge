"""Training losses for the GRU mask model (pure torch).

cmag:   MSE(|S_hat|^c, |S|^c) on the model's STFT magnitudes (gru_v1/v2 loss).
mrstft: multi-resolution STFT loss on time-domain waves; per resolution (FFT 256/512/1024,
        hop 64/128/256, periodic hann) spectral convergence ||  |S|-|S_hat| ||_F / || |S| ||_F
        plus mean |log|S| - log|S_hat||, averaged over resolutions.
"""
from __future__ import annotations

import torch

LOSSES = ("cmag", "cmag+mrstft")
MRSTFT_RES = ((256, 64), (512, 128), (1024, 256))
_EPS = 1e-7


def cmag_loss(est_mag: torch.Tensor, tgt_mag: torch.Tensor, c: float = 0.3) -> torch.Tensor:
    return torch.mean((est_mag.clamp_min(1e-8) ** c - tgt_mag.clamp_min(1e-8) ** c) ** 2)


def _mag(wave: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, periodic=True, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(wave, n_fft, hop, window=win, center=True, pad_mode="reflect",
                      return_complex=True)
    # sqrt(re^2 + im^2 + eps): finite gradient at exact zeros
    return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + _EPS)


def mrstft_loss(est: torch.Tensor, clean: torch.Tensor, resolutions=MRSTFT_RES) -> torch.Tensor:
    """(B, N) estimate vs (B, N) clean -> scalar."""
    total = est.new_zeros(())
    for n_fft, hop in resolutions:
        s_hat, s = _mag(est, n_fft, hop), _mag(clean, n_fft, hop)
        sc = torch.linalg.norm((s - s_hat).flatten(1), dim=1) / \
            torch.linalg.norm(s.flatten(1), dim=1).clamp_min(1e-5)
        total = total + sc.mean() + torch.mean(torch.abs(torch.log(s) - torch.log(s_hat)))
    return total / len(resolutions)
