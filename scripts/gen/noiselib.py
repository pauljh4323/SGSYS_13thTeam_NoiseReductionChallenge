"""Shared helpers for the external-noise generalisation suite (exp/r2-gen).

- list_corpus: recordings of DEMAND / ESC-50 / MUSAN-noise under data_ext/ with their split unit ("group").
- active_noise: active-segment extraction (20 ms frames, energy > -40 dB re the clip's loudest frame; only for
  clips with a long silence, i.e. an inactive run >= 0.5 s); nearly silent clips are dropped.
- fingerprint / match: Haitsma & Kalker (2002) style 32-bit sub-band energy-difference fingerprints, used only
  for the overlap check.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "data_ext"
SR = 16000
CORPORA = ("demand", "esc50", "musan")

FRAME = 320                 # 20 ms activity frames
ACTIVE_DB = -40.0           # re the clip's loudest frame
LONG_SILENCE_S = 0.5        # an inactive run this long triggers active-segment extraction
MIN_ACTIVE_S = 0.25         # less active audio than this -> "nearly silent", dropped
SILENT_DBFS = -60.0         # loudest frame RMS below this -> "nearly silent", dropped
XFADE = 80                  # 5 ms linear crossfade between concatenated active runs


def list_corpus(corpus: str) -> list[dict]:
    """[{corpus, rel (posix, relative to data_ext), group (split unit), cls, fold}] sorted by rel."""
    rows = []
    if corpus == "demand":
        for p in sorted((EXT / "demand").glob("*/ch01.wav")):
            rows.append({"corpus": corpus, "rel": p.relative_to(EXT).as_posix(), "group": p.parent.name,
                         "cls": p.parent.name, "fold": ""})
    elif corpus == "esc50":
        with open(EXT / "esc50" / "esc50.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"corpus": corpus, "rel": f"esc50/audio/{r['filename']}", "group": r["src_file"],
                             "cls": r["category"], "fold": r["fold"]})
    elif corpus == "musan":
        for p in sorted((EXT / "musan" / "noise").rglob("*.wav")):
            rel = p.relative_to(EXT).as_posix()
            rows.append({"corpus": corpus, "rel": rel, "group": rel, "cls": p.parent.name, "fold": ""})
    else:
        raise ValueError(corpus)
    return sorted(rows, key=lambda r: r["rel"])


def read_noise(rel: str) -> np.ndarray:
    x, sr = sf.read(str(EXT / rel), dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    assert sr == SR, (rel, sr)
    return x


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def active_noise(x: np.ndarray) -> tuple[np.ndarray | None, str]:
    """-> (processed noise or None if nearly silent, status in {"kept", "trimmed", "silent"})."""
    n_fr = x.size // FRAME
    if n_fr == 0:
        return None, "silent"
    e = np.mean(x[:n_fr * FRAME].reshape(n_fr, FRAME) ** 2, axis=1)
    emax = float(e.max())
    if emax <= 0 or 10 * np.log10(emax) < SILENT_DBFS:  # loudest 20 ms frame's mean power, dBFS
        return None, "silent"
    act = 10 * np.log10(np.maximum(e, 1e-30) / emax) > ACTIVE_DB
    if act.sum() * FRAME < MIN_ACTIVE_S * SR:
        return None, "silent"
    long_gap = any(b - a >= LONG_SILENCE_S * SR / FRAME for a, b in _runs(~act))
    if not long_gap:
        return x, "kept"
    out = None
    for a, b in _runs(act):
        seg = x[a * FRAME:b * FRAME]
        if out is None:
            out = seg.copy()
        elif seg.size > 2 * XFADE and out.size > 2 * XFADE:
            w = np.linspace(0.0, 1.0, XFADE)
            out[-XFADE:] = out[-XFADE:] * (1 - w) + seg[:XFADE] * w
            out = np.concatenate([out, seg[XFADE:]])
        else:
            out = np.concatenate([out, seg])
    return out, "trimmed"


def loop_excerpt(x: np.ndarray, n: int, offset: int) -> np.ndarray:
    """n samples of x starting at offset, looping (x tiled) if needed."""
    reps = int(np.ceil((offset + n) / x.size)) + 1
    return np.tile(x, reps)[offset:offset + n].copy()


# ---------------------------------------------------------------- fingerprints (overlap check only)
FP_NFFT, FP_HOP = 4096, 256            # 256 ms frames, 16 ms hop
FP_BANDS = np.geomspace(300.0, 3000.0, 34)
FP_GATE_DB = -50.0                     # frames this far below the file's loudest frame are not used


def fingerprint(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sub-fingerprints uint32 (T-1,), valid bool (T-1,))."""
    if x.size < FP_NFFT + FP_HOP:
        return np.zeros(0, np.uint32), np.zeros(0, bool)
    f, _, z = stft(x, SR, window="hann", nperseg=FP_NFFT, noverlap=FP_NFFT - FP_HOP, boundary=None,
                   padded=False)
    p = np.abs(z) ** 2
    idx = np.digitize(f, FP_BANDS) - 1
    e = np.stack([p[idx == b].sum(axis=0) for b in range(33)])          # (33, T)
    d = e[:-1] - e[1:]                                                  # (32, T)
    bits = (d[:, 1:] - d[:, :-1]) > 0                                   # (32, T-1)
    w = (1 << np.arange(32, dtype=np.uint64))
    sub = (bits.T.astype(np.uint64) * w).sum(axis=1).astype(np.uint32)
    tot = e.sum(axis=0)
    valid = 10 * np.log10(np.maximum(tot[1:], 1e-30) / max(tot.max(), 1e-30)) > FP_GATE_DB
    valid &= (sub != 0) & (sub != 0xFFFFFFFF)
    return sub, valid


_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def popcount32(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.uint32)
    return (_POP[a & 0xFF] + _POP[(a >> 8) & 0xFF] + _POP[(a >> 16) & 0xFF] + _POP[a >> 24]).astype(np.int64)


def ber(q: np.ndarray, qv: np.ndarray, r: np.ndarray, rv: np.ndarray) -> float:
    m = qv & rv
    if m.sum() < 32:
        return 1.0
    return float(popcount32(q[m] ^ r[m]).sum() / (32 * m.sum()))
