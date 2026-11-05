"""Submission entry point: enhance every .wav under input/ -> output/ (same relative path).

    python main.py [--input input] [--output output] [--config configs/submission.yaml]

Any channel count / sample rate is accepted: channels are averaged to mono, non-16 kHz audio is
resampled to 16 kHz for enhancement and back to the original rate. Output keeps the input's
filename, sample rate and length and is written as 16-bit PCM. If a file fails, the input is
copied through unchanged and processing continues. CPU-only, offline.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from math import gcd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import yaml  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from nrc.io import write_wav_pcm16  # noqa: E402
from nrc.methods import get_method  # noqa: E402

MODEL_SR = 16000
DEFAULT_CONFIG = REPO_ROOT / "configs" / "submission.yaml"


def load_method(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return get_method(cfg["method"], **(cfg.get("params") or {}))


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if y.size >= n:
        return y[:n]
    return np.pad(y, (0, n - y.size))


def enhance_file(method, src: Path, dst: Path) -> int:
    """Enhance one file; returns the number of clipped samples."""
    x, sr = sf.read(str(src), dtype="float64", always_2d=True)
    n = x.shape[0]
    if n == 0:
        raise ValueError("empty file")
    x = x.mean(axis=1)
    if sr != MODEL_SR:
        g = gcd(MODEL_SR, sr)
        x16 = resample_poly(x, MODEL_SR // g, sr // g)
        y16 = np.asarray(method(x16, MODEL_SR), dtype=np.float64)
        y = resample_poly(y16, sr // g, MODEL_SR // g)
    else:
        y = np.asarray(method(x, MODEL_SR), dtype=np.float64)
    y = _fit_length(y, n)
    if not np.isfinite(y).all():
        raise ValueError("non-finite output")
    return write_wav_pcm16(dst, y, sr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="input")
    parser.add_argument("--output", default="output")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    in_dir, out_dir = Path(args.input).resolve(), Path(args.output).resolve()
    if not in_dir.is_dir():
        print(f"error: input directory not found: {in_dir}", file=sys.stderr)
        return 1
    method = load_method(args.config)
    files = sorted(p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")

    started = time.perf_counter()
    n_failed = n_clipped = 0
    for src in files:
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            n_clipped += enhance_file(method, src, dst)
        except Exception as exc:  # noqa: BLE001  # never abort the batch
            n_failed += 1
            print(f"FAILED {rel}: {type(exc).__name__}: {exc} -> copied input through",
                  file=sys.stderr)
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            except Exception as exc2:  # noqa: BLE001
                print(f"FAILED copy {rel}: {exc2}", file=sys.stderr)
    elapsed = time.perf_counter() - started
    print(f"main.py: processed={len(files)} failed={n_failed} clipped_samples={n_clipped} "
          f"total={elapsed:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
