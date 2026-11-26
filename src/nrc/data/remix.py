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
exp/r2-dmix clean-side augmentation (gru_v8p2z3a/b; defaults off = no draws). Applied to the drawn clean
utterance before mixing, in the dynamic-mixing branch and the remix branch (not to original pairs, whose noisy is
fixed); the target is the augmented clean and the SNR is set against it:
- p_speed (Ko et al. 2015 speed perturbation): with probability p_speed, factor f ~ U(speed_range) (quantised to
  1/200), polyphase resampling by up/down = 200/round(200 f) (scipy resample_poly with a Kaiser FIR: passband to
  0.95, stopband >= 80 dB from 1.0 of the lower Nyquist, so nothing aliases above 8 kHz). Tempo and pitch both scale
  by f. The result is randomly cropped (f < 1) or zero-padded at a random offset (f > 1) back to the original length.
- p_eq (random EQ): with probability p_eq, an RBJ-cookbook low-shelf (corner U(eq_low_hz)) and high-shelf
  (corner U(eq_high_hz)) biquad, shelf slope 1, gains each U(eq_gain_db) dB, cascaded (float64); the clean is then
  rescaled to its original RMS.
exp/r2-ext ext_mix (gru_v8p2z6a/b; default None = off, no draws). Every sample is synthetic (replaces p_dm):
- clean: a random 3 s crop of a LibriSpeech utterance (nrc.data.extpool package) with probability p_libri, else a
  random challenge train clean; then the clean-side augmentation above (speed, EQ), so the target is the augmented
  clean;
- noise: external train_pool noise (corpus uniform, file uniform, random offset, looped if short) with probability
  p_ext_noise, else a challenge train noise (valid, other file than the clean, random circular shift); with
  probability p_second a second noise from a different source, drawn the same way, at U(second_rel_db) dB relative
  to the first; the total noise is scaled to SNR U(snr_db) against the clean; then the peak guard and rms_norm.
All randomness comes from the numpy Generator passed in, so batches are reproducible.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import firwin, kaiserord, resample_poly, sosfilt

from ..io import read_row

RMS_EPS = 1e-6   # = nrc.methods.gru_mask.RMS_EPS (rms_norm denominator floor)
SPEED_DEN = 200   # speed factor resolution 1/200: 41 factors in [0.9, 1.1] (40 cached filters, ~7 MB),
#                   61 in [0.85, 1.15] (gru_v8p2z3d; 60 filters, ~11 MB), 81 in [0.80, 1.20] (gru_v8p2z3e;
#                   80 filters, 14.4 MiB, built in 0.5 s); cache bound 256 filters
SR = 16000


def speed_resample(x: np.ndarray, factor: float) -> np.ndarray:
    """Play x `factor` times faster (tempo and pitch): polyphase resampling to length ~len(x)/factor (float64).
    Exact copy when the quantised factor is 1."""
    up, down = SPEED_DEN, int(round(factor * SPEED_DEN))
    g = gcd(up, down)
    up, down = up // g, down // g
    x = np.asarray(x, np.float64)
    return x.copy() if up == down else resample_poly(x, up, down, window=_aa_filter(up, down))


_AA_CACHE: dict = {}


def _aa_filter(up: int, down: int) -> np.ndarray:
    """Anti-aliasing/imaging low-pass for resample_poly at the upsampled rate: Kaiser FIR, stopband >= 80 dB from the
    lower of the input/output Nyquists, passband to 0.95 of it (scipy's default filter only gives ~16 dB there)."""
    key = (up, down)
    if key not in _AA_CACHE:
        m = max(up, down)
        numtaps, beta = kaiserord(80.0, 0.05 / m)
        numtaps |= 1  # odd: zero-phase delay compensation in resample_poly
        _AA_CACHE[key] = firwin(numtaps, 0.975 / m, window=("kaiser", beta)) * up
        if len(_AA_CACHE) > 256:  # bound for unusual speed_range values
            _AA_CACHE.pop(next(iter(_AA_CACHE)))
    return _AA_CACHE[key]


def fit_length(y: np.ndarray, length: int, offset: int) -> np.ndarray:
    """Crop y to `length` starting at `offset`, or zero-pad it with `offset` leading zeros."""
    if y.size >= length:
        return y[offset:offset + length]
    out = np.zeros(length, y.dtype)
    out[offset:offset + y.size] = y
    return out


def shelf_sos(kind: str, f0: float, gain_db: float, sr: int = SR) -> np.ndarray:
    """RBJ Audio-EQ-Cookbook shelving biquad (shelf slope S = 1) as one normalised sos row."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cw, alpha = np.cos(w0), np.sin(w0) / 2 * np.sqrt(2.0)
    sa = 2 * np.sqrt(A) * alpha
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == "high":
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    return np.array([[*(np.array(b) / a[0]), 1.0, a[1] / a[0], a[2] / a[0]]])


def random_eq(x: np.ndarray, f_low: float, g_low: float, f_high: float, g_high: float) -> np.ndarray:
    """Low-shelf + high-shelf cascade, output rescaled to the input RMS (float64)."""
    x = np.asarray(x, np.float64)
    y = sosfilt(np.vstack([shelf_sos("low", f_low, g_low), shelf_sos("high", f_high, g_high)]), x)
    r_in, r_out = float(np.sqrt(np.mean(x ** 2))), float(np.sqrt(np.mean(y ** 2)))
    return y * (r_in / r_out) if r_out > 0 else y


class RemixData:
    def __init__(self, clean: np.ndarray, noisy: np.ndarray, noise_ok: np.ndarray | None = None,
                 p_original: float = 0.5, snr_db=(-5.0, 20.0), gain_db=None,
                 peak: float = 0.99, gain_db_range=None, apply_to_original_pairs: bool | None = None,
                 gain_mode: str | None = None, p_wide: float = 0.5, gain_wide_db=(-30.0, 6.0),
                 gain_narrow_db=(-6.0, 3.0), rms_norm_target: float | None = None, p_dm: float = 0.0,
                 dm_snr_db=(-5.0, 20.0), dm_p_second: float = 0.2, dm_second_rel_db=(-10.0, 0.0),
                 p_speed: float = 0.0, speed_range=(0.9, 1.1), p_eq: float = 0.0, eq_low_hz=(150.0, 500.0),
                 eq_high_hz=(2000.0, 5000.0), eq_gain_db=(-6.0, 6.0), ext_mix: dict | None = None):
        self.ext = None
        if ext_mix is not None:
            self._init_ext(dict(ext_mix), p_dm)
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
        self.p_speed, self.p_eq = float(p_speed), float(p_eq)
        self.speed_range, self.eq_low_hz = tuple(speed_range), tuple(eq_low_hz)
        self.eq_high_hz, self.eq_gain_db = tuple(eq_high_hz), tuple(eq_gain_db)
        for name, r in (("speed_range", self.speed_range), ("eq_low_hz", self.eq_low_hz),
                        ("eq_high_hz", self.eq_high_hz), ("eq_gain_db", self.eq_gain_db)):
            if len(r) != 2 or r[0] > r[1]:
                raise ValueError(f"{name} must be (lo, hi) with lo <= hi, got {r}")
        if not (0.0 <= self.p_speed <= 1.0 and 0.0 <= self.p_eq <= 1.0) or self.speed_range[0] <= 0 \
                or not (0 < self.eq_low_hz[0] and self.eq_high_hz[1] < SR / 2):
            raise ValueError("p_speed/p_eq must be in [0, 1], speed > 0, shelf corners in (0, 8000) Hz")
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
        if self.ext is not None:
            c, m = self._ext_sample(rng, info)
        elif self.p_dm > 0 and rng.random() < self.p_dm:
            c, m = self._dm_sample(rng, info)
        else:
            c, m = self._remix_sample(rng)
        return (c, m) if self.rms_norm_target is None else self._rms_norm(c, m)

    def _rms_norm(self, c: np.ndarray, mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """g = target / max(rms(mix), RMS_EPS); clean and mix scaled alike (float64 product, float32 out)."""
        m64 = mix.astype(np.float64)
        g = self.rms_norm_target / max(float(np.sqrt(np.mean(m64 ** 2))), RMS_EPS)
        return (c.astype(np.float64) * g).astype(np.float32), (m64 * g).astype(np.float32)

    def _clean_aug(self, c: np.ndarray, rng: np.random.Generator, info: dict | None = None) -> np.ndarray:
        """Speed perturbation then random EQ on a clean utterance (no draws and c returned as is when both off)."""
        if self.p_speed > 0 and rng.random() < self.p_speed:
            f = rng.uniform(*self.speed_range)
            y = speed_resample(c, f)
            off = int(rng.integers(abs(y.size - c.size) + 1))
            c = fit_length(y, c.size, off)
            if info is not None:
                info["speed"] = round(f * SPEED_DEN) / SPEED_DEN
        if self.p_eq > 0 and rng.random() < self.p_eq:
            fl, fh = rng.uniform(*self.eq_low_hz), rng.uniform(*self.eq_high_hz)
            gl, gh = rng.uniform(*self.eq_gain_db), rng.uniform(*self.eq_gain_db)
            c = random_eq(c, fl, gl, fh, gh)
            if info is not None:
                info["eq"] = (fl, gl, fh, gh)
        return c

    def _draw_other(self, rng: np.random.Generator, exclude: set) -> int:
        while True:
            j = int(rng.choice(self.noise_idx))
            if j not in exclude:
                return j

    def _dm_sample(self, rng: np.random.Generator, info: dict | None) -> tuple[np.ndarray, np.ndarray]:
        i = int(rng.integers(len(self)))
        c = self._clean_aug(self.clean[i].astype(np.float64), rng, info)
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

    # ------------------------------------------------------------------ ext_mix (exp/r2-ext, gru_v8p2z6a/b)
    EXT_KEYS = ("package", "pool", "p_libri", "p_ext_noise", "snr_db", "p_second", "second_rel_db", "crop_tries",
                "crop_min_rel_rms")

    def _init_ext(self, cfg: dict, p_dm: float) -> None:
        unknown = set(cfg) - set(self.EXT_KEYS)
        if unknown:
            raise ValueError(f"unknown ext_mix keys: {sorted(unknown)}")
        if float(p_dm) > 0:
            raise ValueError("ext_mix replaces dynamic mixing; set p_dm 0 (or remove it)")
        pool = cfg.get("pool")
        if pool is None:
            from .extpool import ExtPool
            pool = ExtPool(cfg["package"])
        self.ext = pool
        self.ext_p_libri, self.ext_p_noise = float(cfg.get("p_libri", 0.5)), float(cfg.get("p_ext_noise", 0.6))
        self.ext_p_second = float(cfg.get("p_second", 0.3))
        self.ext_snr_db, self.ext_rel_db = tuple(cfg.get("snr_db", (-10.0, 20.0))), tuple(cfg.get("second_rel_db",
                                                                                                    (-10.0, 0.0)))
        self.ext_crop_tries, self.ext_crop_rel = int(cfg.get("crop_tries", 5)), float(cfg.get("crop_min_rel_rms", 0.1))
        for name, p in (("p_libri", self.ext_p_libri), ("p_ext_noise", self.ext_p_noise),
                        ("p_second", self.ext_p_second)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"ext_mix {name} must be in [0, 1], got {p}")
        for name, r in (("snr_db", self.ext_snr_db), ("second_rel_db", self.ext_rel_db)):
            if len(r) != 2 or r[0] > r[1]:
                raise ValueError(f"ext_mix {name} must be (lo, hi) with lo <= hi, got {r}")

    def _libri_crop(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        """Random utterance, random 3 s crop; a crop quieter than crop_min_rel_rms x the utterance RMS is redrawn
        (same utterance, up to crop_tries crops; the last one is kept)."""
        k = int(rng.integers(len(self.ext.clean)))
        u, L = self.ext.clean.get(k), self.length
        thr = self.ext_crop_rel * self.ext.clean_rms[k]
        for _ in range(self.ext_crop_tries):
            o = int(rng.integers(u.size - L + 1))
            c = np.asarray(u[o:o + L], np.float64) / 32768.0
            if np.sqrt(np.mean(c ** 2)) >= thr:
                break
        return c, k

    def _ext_noise(self, rng: np.random.Generator, clean_i: int | None, exclude: tuple | None):
        """One noise excerpt (float64, self.length) and its source key ("ch", j) / ("ext", k)."""
        L = self.length
        for _ in range(8):
            if rng.random() < self.ext_p_noise:
                corp = self.ext.corpora[int(rng.integers(len(self.ext.corpora)))]
                k = int(rng.choice(self.ext.noise_by_corpus[corp]))
                key = ("ext", k)
                if key == exclude:
                    continue
                x = self.ext.noise.get(k)
                idx = (int(rng.integers(x.size)) + np.arange(L)) % x.size  # looping excerpt (= noiselib.loop_excerpt)
                n = np.asarray(x[idx], np.float64) / 32768.0
            else:
                ex = set() if clean_i is None else {clean_i}
                if exclude is not None and exclude[0] == "ch":
                    ex.add(exclude[1])
                j = self._draw_other(rng, ex)
                key = ("ch", j)
                n = np.roll(self.noise[j], int(rng.integers(L))).astype(np.float64)
            if float(np.mean(n ** 2)) > 0:
                return n, key
        raise RuntimeError("ext_mix: 8 silent noise draws in a row")

    def _ext_sample(self, rng: np.random.Generator, info: dict | None) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < self.ext_p_libri:
            c, k = self._libri_crop(rng)
            i, src = None, ("libri", k)
        else:
            i = int(rng.integers(len(self)))
            c, src = self.clean[i].astype(np.float64), ("ch", i)
        c = self._clean_aug(c, rng, info)
        n, n1 = self._ext_noise(rng, i, None)
        n2, rel = None, None
        if rng.random() < self.ext_p_second:
            m2, n2 = self._ext_noise(rng, i, n1)
            rel = rng.uniform(*self.ext_rel_db)
            n = n + m2 * np.sqrt(float(np.mean(n ** 2)) / float(np.mean(m2 ** 2)) * 10 ** (rel / 10))
        snr = rng.uniform(*self.ext_snr_db)
        pc, pn = float(np.mean(c ** 2)), float(np.mean(n ** 2))
        scale = np.sqrt(pc / (pn * 10 ** (snr / 10))) if pn > 0 and pc > 0 else 1.0
        if info is not None:
            info.update(clean_src=src, noise=n1, noise2=n2, rel_db=rel, snr_db=snr)
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
        if self.p_speed > 0 or self.p_eq > 0:
            c = self._clean_aug(c.astype(np.float64), rng).astype(np.float32)
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
