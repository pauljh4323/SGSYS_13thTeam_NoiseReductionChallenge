"""On-the-fly remix augmentation over the train split (RAM-preloaded float32 arrays).

Each sample: with probability p_original the original (clean, noisy) pair (unchanged unless
apply_to_original_pairs); otherwise a random train clean + a random train noise (noise = noisy -
clean, random circular shift and sign) mixed at SNR ~ U(snr_db). Then a random overall gain
~ U(gain_db) (alias gain_db_range) is applied to mixture and target alike, with a peak guard
(both scaled down if |mix| would exceed `peak`). With apply_to_original_pairs=False (default,
gru_v1) original pairs take no gain and consume no extra random draws.
All randomness comes from the numpy Generator passed in, so batches are reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_row


class RemixData:
    def __init__(self, clean: np.ndarray, noisy: np.ndarray, noise_ok: np.ndarray | None = None,
                 p_original: float = 0.5, snr_db=(-5.0, 20.0), gain_db=(-6.0, 3.0),
                 peak: float = 0.99, gain_db_range=None, apply_to_original_pairs: bool = False):
        if clean.shape != noisy.shape or clean.ndim != 2:
            raise ValueError("clean/noisy must be equal-shape (N, L) arrays")
        self.clean = clean.astype(np.float32, copy=False)
        self.noisy = noisy.astype(np.float32, copy=False)
        self.noise = self.noisy - self.clean
        ok = np.ones(len(clean), bool) if noise_ok is None else np.asarray(noise_ok, bool)
        self.noise_idx = np.flatnonzero(ok)
        if self.noise_idx.size == 0:
            raise ValueError("no valid noise sources")
        if gain_db_range is not None:
            gain_db = gain_db_range
        self.p_original, self.snr_db, self.gain_db = float(p_original), tuple(snr_db), tuple(gain_db)
        if len(self.gain_db) != 2 or self.gain_db[0] > self.gain_db[1]:
            raise ValueError(f"gain_db must be (lo, hi) with lo <= hi, got {self.gain_db}")
        self.apply_to_original_pairs = bool(apply_to_original_pairs)
        self.peak = float(peak)

    def __len__(self) -> int:
        return len(self.clean)

    @property
    def length(self) -> int:
        return self.clean.shape[1]

    @classmethod
    def from_manifest(cls, root, df: pd.DataFrame, noise_check_csv=None, **kw) -> "RemixData":
        clean = np.stack([read_row(root, r, "clean").astype(np.float32) for _, r in df.iterrows()])
        noisy = np.stack([read_row(root, r, "noisy").astype(np.float32) for _, r in df.iterrows()])
        noise_ok = None
        if noise_check_csv is not None and Path(noise_check_csv).is_file():
            chk = pd.read_csv(noise_check_csv).set_index("segment_id")["valid"]
            noise_ok = df.segment_id.map(chk).fillna(True).astype(bool).to_numpy()
        return cls(clean, noisy, noise_ok, **kw)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Return one (clean, noisy) float32 pair."""
        if rng.random() < self.p_original:
            i = int(rng.integers(len(self)))
            if not self.apply_to_original_pairs:
                return self.clean[i], self.noisy[i]
            g = 10 ** (rng.uniform(*self.gain_db) / 20)
            return self._peak_guard(self.clean[i] * g, self.noisy[i] * g)
        c = self.clean[int(rng.integers(len(self)))]
        n = self.noise[int(rng.choice(self.noise_idx))]
        n = np.roll(n, int(rng.integers(n.size))) * (1.0 if rng.random() < 0.5 else -1.0)
        pc, pn = float(np.mean(c.astype(np.float64) ** 2)), float(np.mean(n.astype(np.float64) ** 2))
        snr = rng.uniform(*self.snr_db)
        scale = np.sqrt(pc / (pn * 10 ** (snr / 10))) if pn > 0 and pc > 0 else 1.0
        g = 10 ** (rng.uniform(*self.gain_db) / 20)
        c_out = c * g
        return self._peak_guard(c_out, c_out + n * (scale * g))

    def _peak_guard(self, c_out: np.ndarray, mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rescale (clean, mix) together if |mix| would exceed peak."""
        pk = float(np.max(np.abs(mix)))
        if pk > self.peak:
            c_out, mix = c_out * (self.peak / pk), mix * (self.peak / pk)
        return c_out.astype(np.float32), mix.astype(np.float32)

    def batch(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
        pairs = [self.sample(rng) for _ in range(size)]
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
