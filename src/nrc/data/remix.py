"""On-the-fly remix augmentation over the train split (RAM-preloaded float32 arrays).

Each sample: with probability p_original the original (clean, noisy) pair (unchanged unless
apply_to_original_pairs); otherwise a random train clean + a random train noise (noise = noisy -
clean, random circular shift and sign) mixed at SNR ~ U(snr_db). Then a random overall gain
~ U(gain_db) (alias gain_db_range) is applied to mixture and target alike, with a peak guard
(both scaled down if |mix| would exceed `peak`). With apply_to_original_pairs=False (default,
gru_v1) original pairs take no gain and consume no extra random draws.
gain_mode "mixture" (gru_v3): every sample (original pairs included) takes a gain drawn with
probability p_wide from U(gain_wide_db) (default -30..+6 dB), else from U(gain_narrow_db)
(default -6..+3 dB); clean target scaled identically, same peak guard. gain_mode "uniform"
(default) keeps the previous sampling bit-identical.
exp/r2-dmix options (defaults: off, no extra random draws, so the stream above stays bit-identical):
- rms_norm_target (gru_v8p2z1/z2): train-time copy of inference level_mode "rms_norm". Applied last, after the
  peak guard: g = target / max(rms(mix), RMS_EPS) (float64, whole-sample RMS), clean and mix both multiplied by g.
  No input peak cap afterwards (as at inference). The gain augmentation is disabled (no gain draws); passing any
  gain option (gain_db, gain_db_range, gain_mode, apply_to_original_pairs) together with it raises.
- p_dm (gru_v8p2z2): dynamic mixing (Zeghidour & Grangier 2021). With probability p_dm (the draw is made only when
  p_dm > 0): random train clean i + noise of a random other train file j != i (valid noise only, random circular
  shift, no sign flip); with probability dm_p_second a second noise k not in {i, j} (random shift) is added at
  U(dm_second_rel_db) dB relative to the first noise's power; the combined noise is scaled so that the SNR against
  the clean equals U(dm_snr_db); then the peak guard (and rms_norm if on). Neither changes the SNR.
All randomness comes from the numpy Generator passed in, so batches are reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_row

RMS_EPS = 1e-6   # = nrc.methods.gru_mask.RMS_EPS (rms_norm denominator floor)


class RemixData:
    def __init__(self, clean: np.ndarray, noisy: np.ndarray, noise_ok: np.ndarray | None = None,
                 p_original: float = 0.5, snr_db=(-5.0, 20.0), gain_db=None,
                 peak: float = 0.99, gain_db_range=None, apply_to_original_pairs: bool | None = None,
                 gain_mode: str | None = None, p_wide: float = 0.5, gain_wide_db=(-30.0, 6.0),
                 gain_narrow_db=(-6.0, 3.0), rms_norm_target: float | None = None, p_dm: float = 0.0,
                 dm_snr_db=(-5.0, 20.0), dm_p_second: float = 0.2, dm_second_rel_db=(-10.0, 0.0)):
        self.rms_norm_target = None if rms_norm_target is None else float(rms_norm_target)
        if self.rms_norm_target is not None:
            if not self.rms_norm_target > 0:
                raise ValueError(f"rms_norm_target must be > 0, got {rms_norm_target!r}")
            given = [k for k, v in (("gain_db", gain_db), ("gain_db_range", gain_db_range), ("gain_mode", gain_mode),
                                    ("apply_to_original_pairs", apply_to_original_pairs)) if v is not None]
            if given:
                raise ValueError(f"rms_norm_target disables the gain augmentation; remove {given}")
        gain_db = (-6.0, 3.0) if gain_db is None else gain_db
        gain_mode = "uniform" if gain_mode is None else gain_mode
        apply_to_original_pairs = bool(apply_to_original_pairs)
        self.p_dm, self.dm_p_second = float(p_dm), float(dm_p_second)
        self.dm_snr_db, self.dm_second_rel_db = tuple(dm_snr_db), tuple(dm_second_rel_db)
        for name, r in (("dm_snr_db", self.dm_snr_db), ("dm_second_rel_db", self.dm_second_rel_db)):
            if len(r) != 2 or r[0] > r[1]:
                raise ValueError(f"{name} must be (lo, hi) with lo <= hi, got {r}")
        if not (0.0 <= self.p_dm <= 1.0 and 0.0 <= self.dm_p_second <= 1.0):
            raise ValueError(f"p_dm and dm_p_second must be in [0, 1], got {p_dm}, {dm_p_second}")
        if clean.shape != noisy.shape or clean.ndim != 2:
            raise ValueError("clean/noisy must be equal-shape (N, L) arrays")
        self.clean = clean.astype(np.float32, copy=False)
        self.noisy = noisy.astype(np.float32, copy=False)
        self.noise = self.noisy - self.clean
        ok = np.ones(len(clean), bool) if noise_ok is None else np.asarray(noise_ok, bool)
        self.noise_idx = np.flatnonzero(ok)
        if self.noise_idx.size == 0:
            raise ValueError("no valid noise sources")
        if self.p_dm > 0 and self.noise_idx.size < (3 if self.dm_p_second > 0 else 2):
            raise ValueError("dynamic mixing needs at least 3 valid noise sources (2 without a second noise)")
        if gain_db_range is not None:
            gain_db = gain_db_range
        self.p_original, self.snr_db, self.gain_db = float(p_original), tuple(snr_db), tuple(gain_db)
        if len(self.gain_db) != 2 or self.gain_db[0] > self.gain_db[1]:
            raise ValueError(f"gain_db must be (lo, hi) with lo <= hi, got {self.gain_db}")
        if gain_mode not in ("uniform", "mixture"):
            raise ValueError(f"gain_mode must be 'uniform' or 'mixture', got {gain_mode!r}")
        self.gain_mode, self.p_wide = gain_mode, float(p_wide)
        self.gain_wide_db, self.gain_narrow_db = tuple(gain_wide_db), tuple(gain_narrow_db)
        for r in (self.gain_wide_db, self.gain_narrow_db):
            if len(r) != 2 or r[0] > r[1]:
                raise ValueError(f"gain range must be (lo, hi) with lo <= hi, got {r}")
        # mixture mode applies the gain to all samples
        self.apply_to_original_pairs = bool(apply_to_original_pairs) or gain_mode == "mixture"
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

    def sample(self, rng: np.random.Generator, info: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return one (clean, noisy) float32 pair. `info` (tests): dynamic-mixing draws are written into it."""
        if self.p_dm > 0 and rng.random() < self.p_dm:
            c, m = self._dm_sample(rng, info)
        else:
            c, m = self._remix_sample(rng)
        return (c, m) if self.rms_norm_target is None else self._rms_norm(c, m)

    def _rms_norm(self, c: np.ndarray, mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """g = target / max(rms(mix), RMS_EPS); clean and mix scaled alike (float64 product, float32 out)."""
        m64 = mix.astype(np.float64)
        g = self.rms_norm_target / max(float(np.sqrt(np.mean(m64 ** 2))), RMS_EPS)
        return (c.astype(np.float64) * g).astype(np.float32), (m64 * g).astype(np.float32)

    def _draw_other(self, rng: np.random.Generator, exclude: set) -> int:
        while True:
            j = int(rng.choice(self.noise_idx))
            if j not in exclude:
                return j

    def _dm_sample(self, rng: np.random.Generator, info: dict | None) -> tuple[np.ndarray, np.ndarray]:
        i = int(rng.integers(len(self)))
        c = self.clean[i].astype(np.float64)
        j = self._draw_other(rng, {i})
        n = np.roll(self.noise[j], int(rng.integers(self.length))).astype(np.float64)
        k, rel = None, None
        if rng.random() < self.dm_p_second:
            k = self._draw_other(rng, {i, j})
            n2 = np.roll(self.noise[k], int(rng.integers(self.length))).astype(np.float64)
            rel = rng.uniform(*self.dm_second_rel_db)
            p1, p2 = float(np.mean(n ** 2)), float(np.mean(n2 ** 2))
            if p1 > 0 and p2 > 0:
                n = n + n2 * np.sqrt(p1 / p2 * 10 ** (rel / 10))
        snr = rng.uniform(*self.dm_snr_db)
        pc, pn = float(np.mean(c ** 2)), float(np.mean(n ** 2))
        scale = np.sqrt(pc / (pn * 10 ** (snr / 10))) if pn > 0 and pc > 0 else 1.0
        if info is not None:
            info.update(clean=i, noise=j, noise2=k, rel_db=rel, snr_db=snr)
        return self._peak_guard(c, c + n * scale)

    def _remix_sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Original / remix branch (unchanged sampling)."""
        if rng.random() < self.p_original:
            i = int(rng.integers(len(self)))
            if not self.apply_to_original_pairs:
                return self.clean[i], self.noisy[i]
            g = self._draw_gain(rng)
            return self._peak_guard(self.clean[i] * g, self.noisy[i] * g)
        c = self.clean[int(rng.integers(len(self)))]
        n = self.noise[int(rng.choice(self.noise_idx))]
        n = np.roll(n, int(rng.integers(n.size))) * (1.0 if rng.random() < 0.5 else -1.0)
        pc, pn = float(np.mean(c.astype(np.float64) ** 2)), float(np.mean(n.astype(np.float64) ** 2))
        snr = rng.uniform(*self.snr_db)
        scale = np.sqrt(pc / (pn * 10 ** (snr / 10))) if pn > 0 and pc > 0 else 1.0
        g = self._draw_gain(rng)
        c_out = c * g
        return self._peak_guard(c_out, c_out + n * (scale * g))

    def _draw_gain(self, rng: np.random.Generator) -> float:
        """Linear gain; "uniform" draws exactly one uniform (unchanged v1/v2 RNG stream). 1.0 (no draw) with rms_norm."""
        if self.rms_norm_target is not None:
            return 1.0
        if self.gain_mode == "mixture":
            lo, hi = self.gain_wide_db if rng.random() < self.p_wide else self.gain_narrow_db
            return 10 ** (rng.uniform(lo, hi) / 20)
        return 10 ** (rng.uniform(*self.gain_db) / 20)

    def _peak_guard(self, c_out: np.ndarray, mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rescale (clean, mix) together if |mix| would exceed peak."""
        pk = float(np.max(np.abs(mix)))
        if pk > self.peak:
            c_out, mix = c_out * (self.peak / pk), mix * (self.peak / pk)
        return c_out.astype(np.float32), mix.astype(np.float32)

    def batch(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
        pairs = [self.sample(rng) for _ in range(size)]
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
