"""WADA-SNR: waveform-amplitude-distribution-analysis SNR estimate (Kim & Stern, Interspeech 2008), exp/r2-ext.

Model: clean speech amplitude ~ Gamma(shape 0.4) with a random sign, additive Gaussian noise. The statistic
  G = log(mean|x|) - mean(log|x|)
depends only on the SNR, so it is mapped back to the SNR through a table G(SNR). This implementation builds that
table itself, by seeded Monte-Carlo (TABLE_N samples per SNR, -20 .. 100 dB in 0.5 dB steps, seed TABLE_SEED); it
does not copy the published table or any public code. Inputs are normalised by their peak and |x| is floored at
EPS before the log; estimates are clamped to the table range [-20, 100] dB.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

SHAPE = 0.4
SNR_GRID = np.arange(-20.0, 100.0 + 0.25, 0.5)
TABLE_N, TABLE_SEED = 400_000, 2008
EPS = 1e-10


def g_stat(x: np.ndarray) -> float:
    a = np.abs(np.asarray(x, np.float64))
    pk = a.max()
    if pk <= 0:
        return float("nan")
    a = np.maximum(a / pk, EPS)
    return float(np.log(a.mean()) - np.log(a).mean())


@lru_cache(maxsize=1)
def table() -> tuple[np.ndarray, np.ndarray]:
    """(snr_db, G) with G strictly increasing in SNR (checked)."""
    rng = np.random.default_rng(TABLE_SEED)
    s = rng.gamma(SHAPE, 1.0, TABLE_N) * rng.choice((-1.0, 1.0), TABLE_N)
    s /= np.sqrt(np.mean(s ** 2))
    n = rng.standard_normal(TABLE_N)
    n /= np.sqrt(np.mean(n ** 2))
    g = np.array([g_stat(s + n * 10 ** (-snr / 20)) for snr in SNR_GRID])
    # Below ~-10 dB G is flat at the Gaussian value (~0.409) and Monte-Carlo jitter breaks strict monotonicity:
    # take the running maximum there; above -10 dB the table must be strictly increasing.
    g = np.maximum.accumulate(g)
    if not np.all(np.diff(g[SNR_GRID >= -10]) > 0):
        raise AssertionError("WADA table is not monotone above -10 dB; increase TABLE_N")
    return SNR_GRID.copy(), g


def wada_snr(x: np.ndarray) -> float:
    """Estimated SNR in dB (clamped to [-20, 100]); nan for digital silence."""
    g = g_stat(x)
    if not np.isfinite(g):
        return float("nan")
    snr, gt = table()
    return float(np.interp(g, gt, snr))
