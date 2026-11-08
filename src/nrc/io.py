"""Data-root resolution, manifest loading, wav I/O and stratified subsets."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "noise_reduction_challenge"
SAMPLE_RATE = 16000
SPLITS = ("train", "validation")
# Derived held-out splits: name -> folder holding samples.csv (paths relative to that folder).
EXTRA_SPLITS = {"val_remix": REPO_ROOT / "data" / "val_remix"}  # scripts/make_val_remix.py
ALL_SPLITS = SPLITS + tuple(EXTRA_SPLITS)
REQUIRED_COLUMNS = {
    "segment_id", "clean_path", "noisy_path", "sample_rate_hz", "num_samples",
    "noise_category", "noise_collection", "noise_code", "noise_difficulty",
    "target_snr_db", "split",
}
STRATA = ("target_snr_db", "noise_difficulty")


def resolve_data_root(cli_value: str | None = None) -> Path:
    """--data-root > $NRC_DATA_ROOT > <repo>/data/noise_reduction_challenge."""
    value = cli_value or os.environ.get("NRC_DATA_ROOT") or DEFAULT_DATA_ROOT
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    return root


def split_root(root: Path, split: str) -> Path:
    """Folder that a split's manifest paths are relative to (data root, or the derived split dir)."""
    if split not in ALL_SPLITS:
        raise ValueError(f"split must be one of {ALL_SPLITS}")
    return Path(EXTRA_SPLITS.get(split, root)).resolve()


def load_manifest(root: Path, split: str) -> pd.DataFrame:
    """Load and validate <root>/<split>/samples.csv (same checks as the course notebook).

    Derived splits (EXTRA_SPLITS) read <split dir>/samples.csv; read rows with split_root(root, split).
    """
    csv_path = (Path(root).resolve() / split if split in SPLITS else split_root(root, split)) / "samples.csv"
    root = split_root(root, split)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError("Empty manifest or required columns missing.")
    if df.segment_id.duplicated().any():
        raise ValueError("Duplicate segment IDs.")
    if not df["split"].eq(split).all():
        raise ValueError("Manifest split mismatch.")
    for column in ("clean_path", "noisy_path"):
        for relative in df[column]:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"Invalid/missing file: {path}")
    return df


def as_waveform(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite mono waveform.")
    return x


def read_wav(path, expect_sr: int = SAMPLE_RATE, expect_len: int | None = None) -> np.ndarray:
    """Read a mono wav as float64 in [-1, 1); validate sample rate and (optionally) length."""
    x, sr = sf.read(str(path), dtype="float64")
    x = as_waveform(x)
    if sr != expect_sr:
        raise ValueError(f"Expected {expect_sr} Hz, got {sr} Hz: {path}")
    if expect_len is not None and x.size != int(expect_len):
        raise ValueError(f"Length mismatch ({x.size} != {expect_len}): {path}")
    return x


def read_row(root: Path, row, kind: str) -> np.ndarray:
    """Read the clean or noisy file of a manifest row."""
    if kind not in ("clean", "noisy"):
        raise ValueError(kind)
    sr = int(row["sample_rate_hz"])
    return read_wav(Path(root) / row[f"{kind}_path"], expect_sr=sr, expect_len=row["num_samples"])


def to_pcm16(y) -> tuple[np.ndarray, int]:
    """Quantize float audio to int16 with a clipping guard; returns (int16, n_clipped)."""
    y = as_waveform(y)
    scaled = np.round(y * 32768.0)
    n_clipped = int(np.count_nonzero((scaled > 32767) | (scaled < -32768)))
    return np.clip(scaled, -32768, 32767).astype(np.int16), n_clipped


def write_wav_pcm16(path, y, sr: int = SAMPLE_RATE) -> int:
    """Write 16-bit PCM wav; returns the number of clipped samples."""
    pcm, n_clipped = to_pcm16(y)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm, sr, subtype="PCM_16")
    return n_clipped


def stratified_subset(df: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Fixed stratified subset (target_snr_db x noise_difficulty), largest-remainder allocation."""
    if n is None or n >= len(df):
        return df.sort_values("segment_id").reset_index(drop=True)
    if n <= 0:
        raise ValueError("subset size must be positive")
    rng = np.random.default_rng(seed)
    groups = {key: g.sort_values("segment_id") for key, g in df.groupby(list(STRATA), sort=True)}
    keys = list(groups)
    sizes = np.array([len(groups[k]) for k in keys])
    quota = n * sizes / sizes.sum()
    alloc = np.floor(quota).astype(int)
    # Distribute the remainder by largest fractional part (ties: larger stratum, then order).
    order = sorted(range(len(keys)), key=lambda i: (-(quota[i] - alloc[i]), -sizes[i], i))
    for i in order[: n - alloc.sum()]:
        alloc[i] += 1
    picks = []
    for key, k in zip(keys, alloc):
        g = groups[key]
        idx = rng.choice(len(g), size=min(int(k), len(g)), replace=False)
        picks.append(g.iloc[np.sort(idx)])
    return pd.concat(picks).sort_values("segment_id").reset_index(drop=True)
