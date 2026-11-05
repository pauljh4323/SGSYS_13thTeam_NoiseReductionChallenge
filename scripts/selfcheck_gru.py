"""Selfcheck for the gru_mask method with random-init weights (no training data needed).

Checks: param count < 2M; output length == input length for 48000, 47999, short inputs and an
all-zeros input; finite output; 16-bit quantization without clipping.
level_norm on: length preserved, all-zeros input (g = 0, silent output), and scale equivariance
(input x0.1 vs x1: outputs differ only by the 0.1 factor, relative error < 1e-3).
level_mode range: length preserved, all-zeros input silent, an in-range input is bit-identical to
level_norm off, and an out-of-range input is moved to the range edge (g matches the formula).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import _common  # noqa: F401  (sys.path)

from nrc.io import to_pcm16
from nrc.methods import get_method
from nrc.methods.gru_mask import level_gain_range_db, measure_level_dbfs


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
        for measure in ("active", "rms"):
            fn_ln = get_method("gru_mask", weights=str(path), device="cpu", level_norm=True,
                               level_measure=measure)
            for name, x in cases.items():
                y = fn_ln(x, 16000)
                assert y.shape == x.shape and np.isfinite(y).all(), (measure, name)
                if name.startswith("zeros"):
                    assert np.max(np.abs(y)) < 1e-6, (measure, name)
            # speech-like level variation so the "active" measure is not trivially the RMS
            x = cases["48000"] * np.repeat(rng.uniform(0.05, 1.0, 48000 // 1600), 1600)
            y1, y01 = fn_ln(x, 16000), fn_ln(0.1 * x, 16000)
            rel = np.linalg.norm(y01 - 0.1 * y1) / np.linalg.norm(0.1 * y1)
            assert rel < 1e-3, (measure, rel)
            print(f"ok  level_norm {measure:<6} length/zeros OK, x0.1 rel err {rel:.2e}")
        lo, hi = -27.88, -15.27
        fn_r = get_method("gru_mask", weights=str(path), device="cpu", level_norm=True,
                          level_measure="active", level_mode="range", lo_dbfs=lo, hi_dbfs=hi)
        for name, x in cases.items():
            y = fn_r(x, 16000)
            assert y.shape == x.shape and np.isfinite(y).all(), ("range", name)
            if name.startswith("zeros"):
                assert np.max(np.abs(y)) < 1e-6, ("range", name)
        x = cases["48000"] * np.repeat(rng.uniform(0.05, 1.0, 48000 // 1600), 1600)
        mid = 0.5 * (lo + hi)
        x_in = x * 10.0 ** ((mid - measure_level_dbfs(x, "active")) / 20.0)
        assert lo < measure_level_dbfs(x_in, "active") < hi
        assert level_gain_range_db(x_in, "active", lo, hi, 30.0, 12.0) == 0.0
        assert np.array_equal(fn_r(x_in, 16000), fn(x_in, 16000)), "in-range not bit-identical"
        for target, edge in ((lo - 6.0, lo), (hi + 6.0, hi)):
            x_out = x * 10.0 ** ((target - measure_level_dbfs(x, "active")) / 20.0)
            g = level_gain_range_db(x_out, "active", lo, hi, 30.0, 12.0)
            assert abs(g - (edge - measure_level_dbfs(x_out, "active"))) < 1e-9, (target, g)
        print("ok  level_mode range  length/zeros OK, in-range bit-identical to off, edge gains OK")
    print(f"gru_mask selfcheck OK: {n_params:,} params, weights {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
