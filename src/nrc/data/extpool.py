"""External training pools for ext_mix (exp/r2-ext, gru_v8p2z6a/b): LibriSpeech clean + external noise train_pool.

Package layout (built by scripts/gen/build_ext_package.py, uploaded to Drive as one folder):
  <pkg>/librispeech/index.csv + shard_XXX.npy   int16 utterances (speaker, chapter, utt, shard, offset, length, rms, ...)
  <pkg>/noise_pool/index.csv  + shard_XXX.npy   int16 noise recordings (corpus, rel, status, shard, offset, length)
  <pkg>/gen/gen_unseen_dev, gen_multi_dev       selection sets (copies of data/gen)
  <pkg>/MANIFEST.json                           SHA-256 per file
Shards are opened with mmap_mode="r", so only the drawn excerpts are paged in; int16 -> float64 on draw (/ 32768).

Leakage guard: every noise `rel` must be listed in splits/gen/<corpus>_train_pool.txt and in none of the
<corpus>_gen_dev.txt / <corpus>_gen_test.txt lists or splits/gen/excluded.txt; check_noise_rels raises otherwise.
ExtPool runs it on load, and the package builder runs it before writing.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = REPO_ROOT / "splits" / "gen"
CORPORA = ("demand", "esc50", "musan")
SHARD_SAMPLES = 128 * 1024 * 1024  # 256 MiB of int16 per shard


def _read_list(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def split_sets(splits_dir: Path = SPLITS_DIR) -> tuple[set[str], set[str]]:
    """(allowed = union of train_pool lists, forbidden = gen_dev + gen_test + excluded)."""
    allowed, forbidden = set(), _read_list(splits_dir / "excluded.txt")
    for c in CORPORA:
        allowed |= _read_list(splits_dir / f"{c}_train_pool.txt")
        forbidden |= _read_list(splits_dir / f"{c}_gen_dev.txt") | _read_list(splits_dir / f"{c}_gen_test.txt")
    if allowed & forbidden:
        raise AssertionError(f"split lists overlap: {sorted(allowed & forbidden)[:5]}")
    return allowed, forbidden


def check_noise_rels(rels, splits_dir: Path = SPLITS_DIR) -> None:
    """Raise if any noise is a gen_dev / gen_test / excluded file or not in a train_pool list."""
    allowed, forbidden = split_sets(splits_dir)
    rels = set(rels)
    bad = sorted(rels & forbidden)
    if bad:
        raise AssertionError(f"{len(bad)} gen_dev/gen_test/excluded noise file(s) in the training pool, e.g. {bad[:3]}")
    unknown = sorted(rels - allowed)
    if unknown:
        raise AssertionError(f"{len(unknown)} noise file(s) not in any train_pool list, e.g. {unknown[:3]}")


class ShardWriter:
    """Append int16 arrays into shard_XXX.npy files of <= SHARD_SAMPLES samples; rows -> index.csv."""

    def __init__(self, out_dir: Path, fields: list[str], shard_samples: int = SHARD_SAMPLES):
        self.dir, self.fields, self.cap = Path(out_dir), list(fields), int(shard_samples)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows, self.buf, self.fill, self.shard = [], [], 0, 0

    def add(self, x: np.ndarray, **meta) -> None:
        x = np.asarray(x)
        if x.dtype != np.int16 or x.ndim != 1:
            raise ValueError("ShardWriter takes 1-D int16 arrays")
        if self.fill and self.fill + x.size > self.cap:
            self._flush()
        self.rows.append({**meta, "shard": self.shard, "offset": self.fill, "length": x.size})
        self.buf.append(x)
        self.fill += x.size

    def _flush(self) -> None:
        if self.buf:
            np.save(self.dir / f"shard_{self.shard:03d}.npy", np.concatenate(self.buf))
            self.shard, self.buf, self.fill = self.shard + 1, [], 0

    def close(self) -> list[dict]:
        self._flush()
        cols = self.fields + ["shard", "offset", "length"]
        with open(self.dir / "index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.rows)
        return self.rows


class ShardStore:
    """Read side of ShardWriter: rows (list of dicts) and get(k) -> int16 view (memory-mapped)."""

    def __init__(self, d: Path):
        self.dir = Path(d)
        with open(self.dir / "index.csv", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"empty index: {self.dir}")
        self.shard = np.array([int(r["shard"]) for r in self.rows])
        self.offset = np.array([int(r["offset"]) for r in self.rows], np.int64)
        self.length = np.array([int(r["length"]) for r in self.rows], np.int64)
        self._mm = {s: np.load(self.dir / f"shard_{s:03d}.npy", mmap_mode="r") for s in np.unique(self.shard)}
        for k in range(len(self.rows)):
            if self.offset[k] + self.length[k] > self._mm[self.shard[k]].size:
                raise ValueError(f"{self.dir}: row {k} exceeds its shard")

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, k: int) -> np.ndarray:
        o = self.offset[k]
        return self._mm[self.shard[k]][o:o + self.length[k]]


class ExtPool:
    """LibriSpeech utterances + external noise (train_pool only) from a package directory."""

    def __init__(self, package, splits_dir: Path = SPLITS_DIR, min_clean: int = 48000):
        pkg = Path(package)
        if not pkg.is_absolute():
            pkg = REPO_ROOT / pkg
        self.package = pkg
        self.clean = ShardStore(pkg / "librispeech")
        self.noise = ShardStore(pkg / "noise_pool")
        check_noise_rels([r["rel"] for r in self.noise.rows], splits_dir)
        if (self.clean.length < min_clean).any():
            raise ValueError(f"LibriSpeech utterances shorter than {min_clean} samples in the package")
        self.clean_rms = np.array([float(r["rms"]) for r in self.clean.rows])
        corp = np.array([r["corpus"] for r in self.noise.rows])
        self.corpora = tuple(c for c in CORPORA if (corp == c).any())
        self.noise_by_corpus = {c: np.flatnonzero(corp == c) for c in self.corpora}

    def summary(self) -> str:
        h = lambda n: n / 16000 / 3600  # noqa: E731
        return (f"ext pool {self.package}: LibriSpeech {len(self.clean)} utts / "
                f"{len({r['speaker'] for r in self.clean.rows})} speakers / {h(self.clean.length.sum()):.1f} h; noise "
                + ", ".join(f"{c} {len(v)} files {h(self.noise.length[v].sum()):.2f} h"
                            for c, v in self.noise_by_corpus.items()))
