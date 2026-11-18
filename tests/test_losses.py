"""Unit tests for nrc.losses.compressed_spectral_loss (Braun & Tashev 2021 weights, gru_v7a).

Run: python tests/test_losses.py   (also collected by pytest if installed; no extra dependency needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nrc.losses import ccplx_loss, cmag_loss, compressed_spectral_loss  # noqa: E402

W = dict(c=0.3, w_mag=0.7, w_cplx=0.3)


def _spec(seed: int, shape=(2, 257, 40)) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def test_zero_for_identical_inputs():
    s = _spec(0)
    assert compressed_spectral_loss(s, s.clone(), **W).item() == 0.0


def test_finite_for_silence():
    tgt = torch.zeros(2, 257, 40, dtype=torch.complex64)
    for est0 in (torch.zeros_like(tgt), _spec(1)):
        est = est0.clone().requires_grad_(True)
        loss = compressed_spectral_loss(est, tgt, **W)
        loss.backward()
        assert torch.isfinite(loss), loss
        assert torch.isfinite(torch.view_as_real(est.grad)).all()
    assert compressed_spectral_loss(torch.zeros_like(tgt), tgt, **W).item() == 0.0


def test_weights_and_phase_sensitivity():
    est, tgt = _spec(2), _spec(3)
    ref = 0.7 * cmag_loss(est.abs(), tgt.abs(), 0.3) + 0.3 * ccplx_loss(est, tgt, 0.3)
    assert torch.allclose(compressed_spectral_loss(est, tgt, **W), ref)
    # magnitude-only default (gru_v5b term) ignores phase; the complex term does not
    flipped = -tgt
    assert compressed_spectral_loss(flipped, tgt).item() < 1e-10
    assert compressed_spectral_loss(flipped, tgt, **W).item() > 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
