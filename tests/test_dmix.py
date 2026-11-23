"""exp/r2-dmix: dynamic mixing + train-time rms_norm in nrc.data.remix.RemixData (synthetic data, CPU).

Run: python tests/test_dmix.py   (also collected by pytest if installed; no extra dependency needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nrc.data.remix import RemixData  # noqa: E402

N, L = 12, 16000


def _data(seed=1):
    r = np.random.default_rng(seed)
    clean = (r.standard_normal((N, L)) * r.uniform(0.01, 0.2, (N, 1))).astype(np.float32)
    noise = (r.standard_normal((N, L)) * r.uniform(0.005, 0.3, (N, 1))).astype(np.float32)
    return clean, clean + noise


def _snr(c, m):
    c, m = c.astype(np.float64), m.astype(np.float64)
    return 10 * np.log10(np.sum(c ** 2) / np.sum((m - c) ** 2))


def test_dynamic_mixing_snr():
    for rms in (None, 0.079130):
        _check_dm_snr(rms)


def _check_dm_snr(rms):
    d = RemixData(*_data(), p_dm=1.0, dm_snr_db=(-5, 20), dm_p_second=0.2, dm_second_rel_db=(-10, 0),
                  rms_norm_target=rms)
    rng, n_second = np.random.default_rng(0), 0
    for _ in range(100):
        info = {}
        c, m = d.sample(rng, info)
        assert c.dtype == m.dtype == np.float32 and c.shape == m.shape == (L,)
        assert abs(_snr(c, m) - info["snr_db"]) < 0.1
        assert -5 <= info["snr_db"] <= 20
        assert info["noise"] != info["clean"]
        if info["noise2"] is not None:
            n_second += 1
            assert info["noise2"] not in (info["clean"], info["noise"]) and -10 <= info["rel_db"] <= 0
    assert 8 <= n_second <= 35  # Binomial(100, 0.2)
    print(f"  dm rms_norm={rms}: 100 draws, |SNR err| < 0.1 dB, second noise in {n_second}")


def test_rms_norm_output_level():
    clean, noisy = _data()
    t = 0.079130
    d = RemixData(clean, noisy, rms_norm_target=t)
    rng = np.random.default_rng(0)
    for _ in range(50):
        c, m = d.sample(rng)
        assert abs(np.sqrt(np.mean(m.astype(np.float64) ** 2)) / t - 1) < 1e-5
    # original pair (p_original 1): output = g * (clean_i, noisy_i), one g for both
    d1 = RemixData(clean, noisy, p_original=1.0, rms_norm_target=t)
    rng = np.random.default_rng(2)
    c, m = d1.sample(rng)
    r2 = np.random.default_rng(2)
    r2.random()
    i = int(r2.integers(N))
    g = t / np.sqrt(np.mean(noisy[i].astype(np.float64) ** 2))
    assert np.array_equal(m, (noisy[i].astype(np.float64) * g).astype(np.float32))
    assert np.array_equal(c, (clean[i].astype(np.float64) * g).astype(np.float32))
    assert rng.bit_generator.state == r2.bit_generator.state  # no gain draw


def test_rms_norm_rejects_gain_options():
    clean, noisy = _data()
    for kw in ({"gain_db": (-40, 6)}, {"gain_mode": "uniform"}, {"apply_to_original_pairs": True},
               {"gain_db_range": (-6, 3)}):
        try:
            RemixData(clean, noisy, rms_norm_target=0.08, **kw)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {kw}")


def test_options_off_draws_unchanged():
    """Defaults draw the same stream as explicitly passing the old defaults."""
    clean, noisy = _data()
    a = RemixData(clean, noisy, gain_db=(-40, 6), apply_to_original_pairs=True)
    b = RemixData(clean, noisy, gain_db=(-40, 6), apply_to_original_pairs=True, p_dm=0.0, rms_norm_target=None)
    ra, rb = np.random.default_rng(0), np.random.default_rng(0)
    for _ in range(50):
        (ca, ma), (cb, mb) = a.sample(ra), b.sample(rb)
        assert np.array_equal(ca, cb) and np.array_equal(ma, mb)
    assert ra.bit_generator.state == rb.bit_generator.state


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
