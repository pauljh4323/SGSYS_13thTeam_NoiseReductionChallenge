"""Training losses for the GRU mask model (pure torch).

cmag:   MSE(|S_hat|^c, |S|^c) on the model's STFT magnitudes (gru_v1/v2 loss).
mrstft: multi-resolution STFT loss on time-domain waves; per resolution (FFT 256/512/1024,
        hop 64/128/256, periodic hann) spectral convergence ||  |S|-|S_hat| ||_F / || |S| ||_F
        plus mean |log|S| - log|S_hat||, averaged over resolutions.
sisdr:  negative scale-invariant SDR (dB) on zero-mean time-domain waves, mean over batch;
        eps-stabilized so an all-zeros clean or estimate stays finite (additive term, gru_v4b).
compressed spectral (Braun & Tashev 2021): w_mag * cmag + w_cplx * ccplx with power-law c; the paper's
        alpha-blend (alpha complex + (1 - alpha) magnitude, c = alpha = 0.3) is w_cplx = alpha, w_mag = 1 - alpha
        (train.lambda_ccplx / train.lambda_cmag, gru_v7a). w_mag = 1, w_cplx = 0 is the gru_v1..v5b cmag term.
"""
from __future__ import annotations

import torch

LOSSES = ("cmag", "cmag+mrstft", "cmag+mrstft+sisdr")
MRSTFT_RES = ((256, 64), (512, 128), (1024, 256))
_EPS = 1e-7


def cmag_loss(est_mag: torch.Tensor, tgt_mag: torch.Tensor, c: float = 0.3) -> torch.Tensor:
    return torch.mean((est_mag.clamp_min(1e-8) ** c - tgt_mag.clamp_min(1e-8) ** c) ** 2)


def ccplx_loss(est_spec: torch.Tensor, tgt_spec: torch.Tensor, c: float = 0.3, eps: float = 1e-8) -> torch.Tensor:
    """Compressed complex spectral MSE: |X|^c e^{j angle X} for estimate and target (off unless lambda_ccplx > 0)."""
    def comp(z):
        mag = z.abs().clamp_min(eps)
        return z * (mag ** (c - 1.0))
    return (comp(est_spec) - comp(tgt_spec)).abs().pow(2).mean()


def compressed_spectral_loss(est_spec: torch.Tensor, tgt_spec: torch.Tensor, c: float = 0.3,
                             w_mag: float = 1.0, w_cplx: float = 0.0,
                             est_mag: torch.Tensor | None = None) -> torch.Tensor:
    """w_mag * cmag(|est|, |tgt|) + w_cplx * ccplx(est, tgt). est_mag overrides |est_spec| (magnitude-mask
    training uses m * |X|, exactly as gru_v5b); the complex term is skipped when w_cplx == 0."""
    loss = w_mag * cmag_loss(est_spec.abs() if est_mag is None else est_mag, tgt_spec.abs(), c)
    if w_cplx > 0:
        loss = loss + w_cplx * ccplx_loss(est_spec, tgt_spec, c)
    return loss


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


def sisdr_loss(est: torch.Tensor, clean: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B, N) estimate vs (B, N) clean -> scalar -mean_b SI-SDR_b (dB)."""
    est = est - est.mean(dim=-1, keepdim=True)
    clean = clean - clean.mean(dim=-1, keepdim=True)
    alpha = (est * clean).sum(-1, keepdim=True) / ((clean ** 2).sum(-1, keepdim=True) + eps)
    target = alpha * clean
    ratio = ((target ** 2).sum(-1) + eps) / (((est - target) ** 2).sum(-1) + eps)
    return -torch.mean(10 * torch.log10(ratio))
