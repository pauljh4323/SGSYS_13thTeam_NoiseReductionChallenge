"""Median whole-file RMS of the train-split noisy files (target_rms of gru_mask level_mode "rms_norm").

Usage: python scripts/rms_target.py [--data-root DIR] > logs/rms_target.log
Computed once; the value is stored in configs/dl/*_rmsn.yaml (params.target_rms).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nrc.io import load_manifest, read_row, resolve_data_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    root = resolve_data_root(a.data_root)
    df = load_manifest(root, "train")
    rms = np.array([np.sqrt(np.mean(read_row(root, r, "noisy").astype(np.float64) ** 2)) for _, r in df.iterrows()])
    med = float(np.median(rms))
    print(f"train noisy files: {rms.size}")
    print(f"RMS p5 / median / p95: {np.percentile(rms, 5):.6f} / {med:.6f} / {np.percentile(rms, 95):.6f}")
    print(f"median dBFS: {20 * np.log10(med):.2f}")
    print(f"target_rms: {med:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
