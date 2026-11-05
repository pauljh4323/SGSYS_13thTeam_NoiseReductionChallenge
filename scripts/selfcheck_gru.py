"""Selfcheck for the gru_mask method with random-init weights (no training data needed).

Checks: param count < 2M; output length == input length for 48000, 47999, short inputs and an
all-zeros input; finite output; 16-bit quantization without clipping.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import _common  # noqa: F401  (sys.path)

from nrc.io import to_pcm16
from nrc.methods import get_method


def main() -> int:
    import torch
    from nrc.models.gru_mask import MAX_PARAMS, GRUMask, count_params, save_weights

    torch.manual_seed(0)
    model = GRUMask()
    n_params = count_params(model)
    assert n_params < MAX_PARAMS, n_params
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "random.pt"
        save_weights(path, model)
        size_mb = path.stat().st_size / 1e6
        fn = get_method("gru_mask", weights=str(path), device="cpu")
        cases = {"48000": 0.1 * rng.standard_normal(48000), "47999": 0.1 * rng.standard_normal(47999),
                 "zeros48000": np.zeros(48000), "short300": 0.1 * rng.standard_normal(300)}
        for name, x in cases.items():
            y = fn(x, 16000)
            assert y.shape == x.shape, (name, y.shape, x.shape)
            assert np.isfinite(y).all(), name
            _, n_clip = to_pcm16(y)
            assert n_clip == 0, (name, n_clip)
            if name.startswith("zeros"):
                assert np.max(np.abs(y)) < 1e-6, name
            print(f"ok  {name:<11} len {y.size}  peak {np.max(np.abs(y)):.4f}")
    print(f"gru_mask selfcheck OK: {n_params:,} params, weights {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
