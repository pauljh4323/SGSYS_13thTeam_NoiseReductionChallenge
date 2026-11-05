"""Sanity-check additive noise extraction on the train split: noise = noisy - clean.

Per pair: SNR(clean, noise) vs target_snr_db, clipping (|noisy| >= 0.999), and validity flags.
Writes logs/noise_check.csv and prints a 5-line summary. Used to decide which pairs may serve
as noise sources for remix augmentation (src/nrc/data/remix.py reads the `valid` column).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT

from nrc.io import load_manifest, read_row, resolve_data_root

CLIP_LEVEL = 0.999
MAX_SNR_ERR_DB = 1.0     # |SNR(clean, noise) - target| above this -> invalid
MAX_CORR = 0.2           # |corr(clean, noise)| above this suggests gain/alignment mismatch
MIN_NOISE_RMS = 1e-5     # practically silent noise -> invalid


def check_pair(clean: np.ndarray, noisy: np.ndarray, target: float) -> dict:
    noise = noisy - clean
    pc, pn = float(np.sum(clean ** 2)), float(np.sum(noise ** 2))
    snr = 10 * np.log10(pc / pn) if pn > 0 and pc > 0 else np.nan
    denom = np.sqrt(pc * pn)
    corr = float(np.dot(clean, noise) / denom) if denom > 0 else np.nan
    n_clip = int(np.count_nonzero(np.abs(noisy) >= CLIP_LEVEL))
    rms = float(np.sqrt(pn / noise.size))
    reasons = []
    if not np.isfinite(snr):
        reasons.append("snr_undefined")
    elif abs(snr - target) > MAX_SNR_ERR_DB:
        reasons.append("snr_mismatch")
    if rms < MIN_NOISE_RMS:
        reasons.append("silent_noise")
    if np.isfinite(corr) and abs(corr) > MAX_CORR:
        reasons.append("clean_noise_corr")
    if n_clip:
        reasons.append("clipped")
    return {"snr_db": snr, "snr_err_db": snr - target if np.isfinite(snr) else np.nan,
            "corr_clean_noise": corr, "noise_rms": rms, "n_clipped": n_clip,
            "valid": not reasons, "reasons": ";".join(reasons)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default=str(REPO_ROOT / "logs" / "noise_check.csv"))
    args = parser.parse_args(argv)
    root = resolve_data_root(args.data_root)
    df = load_manifest(root, "train")
    rows = []
    for _, row in df.iterrows():
        res = check_pair(read_row(root, row, "clean"), read_row(root, row, "noisy"),
                         float(row["target_snr_db"]))
        rows.append({"segment_id": row["segment_id"], "noise_code": row["noise_code"],
                     "target_snr_db": row["target_snr_db"], **res})
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    err = out.snr_err_db.abs()
    print(f"pairs: {len(out)}  ->  {args.out}")
    print(f"SNR error |dB|: mean {err.mean():.4f}, max {err.max():.4f} (signed mean {out.snr_err_db.mean():+.4f})")
    print(f"clipping (|noisy|>={CLIP_LEVEL}): {int((out.n_clipped > 0).sum())} pairs, {int(out.n_clipped.sum())} samples")
    print(f"|corr(clean, noise)|: mean {out.corr_clean_noise.abs().mean():.4f}, max {out.corr_clean_noise.abs().max():.4f}")
    bad = out[~out.valid]
    print(f"invalid pairs: {len(bad)}" + (f" ({bad.reasons.value_counts().to_dict()})" if len(bad) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
