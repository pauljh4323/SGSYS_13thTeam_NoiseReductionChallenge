"""exp/r2-dmix: clean-side augmentation in nrc.data.remix (speed perturbation, random EQ); synthetic data, CPU.

Run: python tests/test_caug.py   (also collected by pytest if installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nrc.data.remix import RemixData, fit_length, random_eq, shelf_sos, speed_resample  # noqa: E402

SR, L = 16000, 48000


def _speech_like(seed=0):
    r = np.random.default_rng(seed)
    return r.standard_normal(L) * 0.05 * (1 + np.sin(2 * np.pi * 3 * np.arange(L) / SR))


def _peak_hz(x):
    w = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * w, 8 * x.size))
    return np.fft.rfftfreq(8 * x.size, 1 / SR)[int(np.argmax(spec))]


def test_speed_identity():
    x = _speech_like()
    y = fit_length(speed_resample(x, 1.0), L, 0)
    err = float(np.max(np.abs(y - x)))
    assert y.shape == x.shape and err < 1e-6, err
    print(f"speed 1.0 identity: max |diff| {err:.1e}")


def test_eq_identity():
    x = _speech_like()
    y = random_eq(x, 300.0, 0.0, 3000.0, 0.0)
    err = float(np.max(np.abs(y - x)))
    assert err < 1e-6, err
    print(f"EQ gains 0 identity: max |diff| {err:.1e}")


def test_tone_and_length():
    t = np.arange(L) / SR
    x = np.sin(2 * np.pi * 1000 * t)
    for f in (1.1, 0.9, 1.15, 0.85):  # 0.85 / 1.15 = gru_v8p2z3d range ends
        y = speed_resample(x, f)
        assert abs(y.size - L / f) <= 1, (y.size, L / f)
        out = fit_length(y, L, (abs(y.size - L)) // 2)
        assert out.size == L
        hz = _peak_hz(y)
        assert abs(hz - 1000 * f) < 2, hz
        print(f"1 kHz tone x {f}: peak {hz:.1f} Hz, resampled length {y.size}, fitted length {out.size}")


def test_no_alias():
    """Speed-up 1.1 of a 7.8 kHz tone (would land at 8.58 kHz, aliasing to 7.42 kHz): removed by the low-pass."""
    t = np.arange(L) / SR
    x = np.sin(2 * np.pi * 7800 * t)
    for f in (1.1, 1.15):  # 1.15 (gru_v8p2z3d): would land at 8.97 kHz, aliasing to 7.03 kHz
        y = speed_resample(x, f)
        rms_ratio = float(np.sqrt(np.mean(y[1000:-1000] ** 2)) / np.sqrt(np.mean(x ** 2)))
        assert rms_ratio < 1e-3, rms_ratio
        print(f"7.8 kHz tone x {f} (above new Nyquist): residual {20 * np.log10(rms_ratio):.1f} dB")


def test_wide_filter_bank():
    """gru_v8p2z3d range [0.85, 1.15]: every quantised factor builds a filter (60 + identity) within the cache bound."""
    from nrc.data import remix as R
    R._AA_CACHE.clear()
    x = np.zeros(L)
    for k in range(170, 231):
        y = speed_resample(x, k / 200)
        assert abs(y.size - L * 200 / k) <= 1
    n, mb = len(R._AA_CACHE), sum(v.nbytes for v in R._AA_CACHE.values()) / 1e6
    assert n == 60, n
    print(f"speed U[0.85, 1.15]: 61 factors, {n} cached filters, {mb:.1f} MB")


def test_shelf_gains():
    from scipy.signal import sosfreqz
    for kind, f0, probe in (("low", 300.0, 20.0), ("high", 3000.0, 7900.0)):
        for g in (-6.0, 6.0):
            _, h = sosfreqz(shelf_sos(kind, f0, g), worN=[probe], fs=SR)
            gd = 20 * np.log10(abs(h[0]))
            assert abs(gd - g) < 0.3, (kind, g, gd)
            _, hc = sosfreqz(shelf_sos(kind, f0, g), worN=[f0], fs=SR)
            assert abs(20 * np.log10(abs(hc[0])) - g / 2) < 0.1  # half gain at the corner (RBJ)
    print("shelf gains at DC / Nyquist side = +-6 dB, half gain at the corner: OK")


def test_eq_rms_and_pipeline():
    r = np.random.default_rng(1)
    clean = (r.standard_normal((12, L)) * r.uniform(0.01, 0.2, (12, 1))).astype(np.float32)
    noisy = clean + (r.standard_normal((12, L)) * 0.05).astype(np.float32)
    x = clean[0].astype(np.float64)
    y = random_eq(x, 200.0, 6.0, 4000.0, -6.0)
    assert abs(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2)) - 1) < 1e-9
    d = RemixData(clean, noisy, p_dm=1.0, rms_norm_target=0.07913, p_speed=0.5, p_eq=0.5)
    rng, n_speed, n_eq = np.random.default_rng(0), 0, 0
    for _ in range(100):
        info = {}
        c, m = d.sample(rng, info)
        assert c.shape == (L,) and m.shape == (L,) and np.all(np.isfinite(c)) and np.all(np.isfinite(m))
        cc, mm = c.astype(np.float64), m.astype(np.float64)
        snr = 10 * np.log10(np.sum(cc ** 2) / np.sum((mm - cc) ** 2))
        assert abs(snr - info["snr_db"]) < 0.1, (snr, info)
        n_speed += "speed" in info
        n_eq += "eq" in info
        if "speed" in info:
            assert 0.9 <= info["speed"] <= 1.1
    assert 30 < n_speed < 70 and 30 < n_eq < 70, (n_speed, n_eq)
    print(f"pipeline (dmix + rms_norm + speed + EQ), 100 draws: lengths OK, SNR vs augmented clean < 0.1 dB, "
          f"speed {n_speed}/100, EQ {n_eq}/100")
    # gru_v8p2z3d settings: p 0.7, speed U[0.85, 1.15], EQ gains U[-9, +9] dB
    d = RemixData(clean, noisy, p_dm=1.0, rms_norm_target=0.07913, p_speed=0.7, speed_range=(0.85, 1.15),
                  p_eq=0.7, eq_gain_db=(-9.0, 9.0))
    rng, n_speed, n_eq, speeds, gains = np.random.default_rng(0), 0, 0, [], []
    for _ in range(100):
        info = {}
        c, m = d.sample(rng, info)
        assert c.shape == (L,) and np.all(np.isfinite(c)) and np.all(np.isfinite(m))
        cc, mm = c.astype(np.float64), m.astype(np.float64)
        assert abs(10 * np.log10(np.sum(cc ** 2) / np.sum((mm - cc) ** 2)) - info["snr_db"]) < 0.1
        if "speed" in info:
            n_speed += 1
            speeds.append(info["speed"])
        if "eq" in info:
            n_eq += 1
            gains += [info["eq"][1], info["eq"][3]]  # info["eq"] = (f_low, g_low, f_high, g_high)
    assert 0.85 <= min(speeds) and max(speeds) <= 1.15 and 55 < n_speed < 85 and 55 < n_eq < 85, (n_speed, n_eq)
    assert max(abs(g) for g in gains) <= 9.0 and max(abs(g) for g in gains) > 6.0
    print(f"pipeline z3d settings, 100 draws: SNR < 0.1 dB, speed {n_speed}/100 in [{min(speeds):.3f}, "
          f"{max(speeds):.3f}], EQ {n_eq}/100, max |gain| {max(abs(g) for g in gains):.2f} dB")


def test_off_no_draws():
    r = np.random.default_rng(2)
    clean = (r.standard_normal((12, L)) * 0.1).astype(np.float32)
    noisy = clean + (r.standard_normal((12, L)) * 0.05).astype(np.float32)
    a = RemixData(clean, noisy, p_dm=1.0, rms_norm_target=0.07913)
    b = RemixData(clean, noisy, p_dm=1.0, rms_norm_target=0.07913, p_speed=0.0, p_eq=0.0)
    ra, rb = np.random.default_rng(0), np.random.default_rng(0)
    for _ in range(50):
        for x, y in zip(a.sample(ra), b.sample(rb)):
            assert np.array_equal(x, y)
    assert ra.bit_generator.state == rb.bit_generator.state
    print("options off (p_speed = p_eq = 0): identical samples and RNG state")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_caug: ALL PASS")
