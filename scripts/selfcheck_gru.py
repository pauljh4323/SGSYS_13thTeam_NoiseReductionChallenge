"""Selfcheck for the gru_mask method with random-init weights (no training data needed).

Checks: param count < 2M; output length == input length for 48000, 47999, short inputs and an
all-zeros input; finite output; 16-bit quantization without clipping.
level_norm on: length preserved, all-zeros input (g = 0, silent output), and scale equivariance
(input x0.1 vs x1: outputs differ only by the 0.1 factor, relative error < 1e-3).
level_mode range: length preserved, all-zeros input silent, an in-range input is bit-identical to
level_norm off, and an out-of-range input is moved to the range edge (g matches the formula).
level_mode range_boost: length preserved, all-zeros input silent, an in-range input is bit-identical
to level_norm off; below lo g = target - measured, above hi g = hi - measured.
front_end (gru_v2): weights/gru_v1.pt rebuilt with explicit front_end="none" is bit-identical to
load_weights (same state_dict keys, model_cfg unchanged); v2b (conv2d, H=256) random init keeps
length for 48000/47999 and gives finite, silent output for all-zeros input.
remix: apply_to_original_pairs off returns original pairs untouched; on, the clean/noisy ratio is
preserved and |noisy| <= peak; gain_mode mixture keeps the ratio, gains stay in U(-30,+6) dB and
~62.5 % fall in [-6,+3] dB (0.5 + 0.5 * 9/36).
losses (gru_v3): mrstft(x, x) ~ 0, mrstft/cmag finite with finite gradients through the ISTFT.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import _common  # noqa: F401  (sys.path)

from nrc.io import to_pcm16
from nrc.methods import get_method
from nrc.methods.gru_mask import (level_gain_range_boost_db, level_gain_range_db,
                                   measure_level_dbfs)


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
        tgt = -19.60
        fn_rb = get_method("gru_mask", weights=str(path), device="cpu", level_norm=True,
                           level_measure="active", level_mode="range_boost", lo_dbfs=lo,
                           hi_dbfs=hi, target_dbfs=tgt)
        for name, xc in cases.items():
            y = fn_rb(xc, 16000)
            assert y.shape == xc.shape and np.isfinite(y).all(), ("range_boost", name)
            if name.startswith("zeros"):
                assert np.max(np.abs(y)) < 1e-6, ("range_boost", name)
        assert level_gain_range_boost_db(x_in, "active", lo, hi, tgt, 30.0, 12.0) == 0.0
        assert np.array_equal(fn_rb(x_in, 16000), fn(x_in, 16000)), "rb in-range not bit-identical"
        for lvl, ref in ((lo - 6.0, tgt), (hi + 6.0, hi)):
            x_out = x * 10.0 ** ((lvl - measure_level_dbfs(x, "active")) / 20.0)
            g = level_gain_range_boost_db(x_out, "active", lo, hi, tgt, 30.0, 12.0)
            assert abs(g - (ref - measure_level_dbfs(x_out, "active"))) < 1e-9, (lvl, g)
        print("ok  level_mode range_boost  length/zeros OK, in-range bit-identical to off, gains OK")
    _check_front_end(rng)
    _check_remix()
    _check_losses()
    print(f"gru_mask selfcheck OK: {n_params:,} params, weights {size_mb:.2f} MB")
    return 0


def _check_front_end(rng) -> None:
    import torch
    from nrc.models.gru_mask import GRUMask, count_params, load_weights

    v1 = _common.REPO_ROOT / "weights" / "gru_v1.pt"
    if v1.is_file():
        ck = torch.load(v1, map_location="cpu", weights_only=True)
        assert set(ck["model_cfg"]) == {"n_fft", "hop", "hidden", "layers"}, ck["model_cfg"]
        ref = load_weights(v1)
        m = GRUMask(**ck["model_cfg"], front_end="none")
        m.load_state_dict(ck["model_state"])  # strict: same key set
        m.eval()
        assert m.config == ref.config, (m.config, ref.config)
        x = torch.from_numpy((0.1 * rng.standard_normal((2, 48000))).astype(np.float32))
        with torch.inference_mode():
            assert torch.equal(m(x), ref(x)), "gru_v1 front_end none not bit-identical"
        print("ok  gru_v1 weights front_end=none bit-identical")
    else:
        print("skip gru_v1 bit-identity (weights/gru_v1.pt missing)")
    torch.manual_seed(0)
    v2b = GRUMask(hidden=256, front_end="conv2d", conv_channels=16, conv_norm="group").eval()
    for n in (48000, 47999):
        for name, x in (("rand", 0.1 * rng.standard_normal(n)), ("zeros", np.zeros(n))):
            with torch.inference_mode():
                y = v2b(torch.from_numpy(x.astype(np.float32))[None])[0].numpy()
            assert y.shape == (n,) and np.isfinite(y).all(), (n, name)
            if name == "zeros":
                assert np.max(np.abs(y)) < 1e-6, n
    print(f"ok  v2b conv2d H=256 random init ({count_params(v2b):,} params) length 48000/47999, zeros")


def _check_remix() -> None:
    from nrc.data.remix import RemixData

    r = np.random.default_rng(0)
    c = (0.3 * r.standard_normal((8, 4000))).astype(np.float32)
    n = c + (0.3 * r.standard_normal((8, 4000))).astype(np.float32)
    off = RemixData(c, n, p_original=1.0)
    co, no = off.sample(np.random.default_rng(1))
    assert any(np.array_equal(co, c[i]) and np.array_equal(no, n[i]) for i in range(8))
    on = RemixData(c, n, p_original=1.0, gain_db_range=(-30, 6), apply_to_original_pairs=True)
    g = np.random.default_rng(2)
    for _ in range(50):
        cs, ns = on.sample(g)
        i = int(np.argmin([np.abs(ns / n[k] - ns[0] / n[k][0]).max() for k in range(8)]))
        ratio = ns[0] / n[i][0]
        assert np.allclose(cs, c[i] * ratio, rtol=1e-4, atol=1e-6) and np.allclose(ns, n[i] * ratio, rtol=1e-4, atol=1e-6)
        assert np.max(np.abs(ns)) <= 0.99 + 1e-6 and ratio <= 10 ** (6 / 20) + 1e-6
    print("ok  remix gain on original pairs (same gain on clean/noisy, peak <= 0.99); off untouched")
    small_c, small_n = 0.01 * c, 0.01 * n
    mix = RemixData(small_c, small_n, p_original=1.0, gain_mode="mixture")
    g, gains = np.random.default_rng(3), []
    for _ in range(2000):
        cs, ns = mix.sample(g)
        i = int(np.argmin([np.abs(ns / small_n[k] - ns[0] / small_n[k][0]).max() for k in range(8)]))
        ratio = ns[0] / small_n[i][0]
        assert np.allclose(cs, small_c[i] * ratio, rtol=1e-4, atol=1e-8)
        gains.append(20 * np.log10(ratio))
    gains = np.array(gains)
    narrow = float(np.mean((gains >= -6 - 1e-4) & (gains <= 3 + 1e-4)))
    assert gains.min() >= -30 - 1e-4 and gains.max() <= 6 + 1e-4, (gains.min(), gains.max())
    assert 0.58 < narrow < 0.67, narrow
    print(f"ok  remix gain_mode mixture (range [{gains.min():.1f}, {gains.max():.1f}] dB, "
          f"in [-6,+3]: {100 * narrow:.1f} %)")


def _check_losses() -> None:
    import torch
    from nrc.losses import cmag_loss, mrstft_loss
    from nrc.models.gru_mask import GRUMask

    torch.manual_seed(0)
    clean = 0.1 * torch.randn(2, 16000)
    noisy = clean + 0.05 * torch.randn(2, 16000)
    assert float(mrstft_loss(clean, clean)) < 1e-6
    model = GRUMask(hidden=32, layers=1)
    spec = model.stft(noisy)
    m = model.mask(spec.abs())
    est = model.istft(m * spec, length=16000)
    loss = cmag_loss(m * spec.abs(), model.stft(clean).abs()) + 0.5 * mrstft_loss(est, clean)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters()
                                        if p.grad is not None)
    print(f"ok  losses cmag+mrstft finite ({float(loss):.4f}), mrstft(x, x) ~ 0, finite grads")


if __name__ == "__main__":
    raise SystemExit(main())
