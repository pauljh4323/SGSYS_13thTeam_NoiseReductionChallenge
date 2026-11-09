"""Input-SNR estimator check (noisy input only): snr_hat_db per file vs target_snr_db.

snr_hat = mmse_lsa.estimate_input_snr_db on the STFT of the noisy file with the noise PSD from the
lowest-energy frames (noise_est "initial", before noise_bias), STFT settings from --config.
Writes logs/snr_hat_<split>_sub<N>.csv; prints correlation, per-target medians and a scale check.
Clean audio is not read.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from _common import REPO_ROOT, load_config

from nrc.io import load_manifest, read_row, resolve_data_root, split_root, stratified_subset
from nrc.methods.mmse_lsa import estimate_input_snr_db
from nrc.methods.wiener import estimate_noise_power
from nrc.stft import stft


def snr_hat_db(x, sr, n_fft, hop, window, noise_fraction):
    _, times, z = stft(x, sr, n_fft, hop, window)
    noise = estimate_noise_power(z, times, len(x), sr, n_fft, noise_fraction)[:, None]
    return estimate_input_snr_db(np.abs(z) ** 2, noise)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dsp_edge.yaml")
    ap.add_argument("--split", default="train", choices=("train", "validation", "val_remix"))
    ap.add_argument("--subset", type=int, default=400)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)
    p = load_config(args.config)["params"]
    st = (p.get("n_fft", 512), p.get("hop", 128), p.get("window", "hann"),
          p.get("noise_fraction", 0.2))
    data_root = resolve_data_root(args.data_root)
    manifest = stratified_subset(load_manifest(data_root, args.split), args.subset)
    root = split_root(data_root, args.split)
    rows, max_scale_err = [], 0.0
    for i, (_, r) in enumerate(manifest.iterrows()):
        x = read_row(root, r.to_dict(), "noisy")
        s = snr_hat_db(x, int(r["sample_rate_hz"]), *st)
        if i < 20:  # scale invariance spot check
            for g in (1e-3, 10.0):
                max_scale_err = max(max_scale_err,
                                    abs(snr_hat_db(g * x, int(r["sample_rate_hz"]), *st) - s))
        rows.append({"segment_id": r["segment_id"], "target_snr_db": r["target_snr_db"],
                     "noise_difficulty": r["noise_difficulty"], "snr_hat_db": s})
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "logs" / f"snr_hat_{args.split}_sub{args.subset}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"n={len(df)} -> {out.relative_to(REPO_ROOT)}")
    print(f"pearson r = {df.snr_hat_db.corr(df.target_snr_db):.4f}, "
          f"spearman r = {df.snr_hat_db.corr(df.target_snr_db, method='spearman'):.4f}")
    print(df.groupby("target_snr_db").snr_hat_db.describe(percentiles=[.25, .5, .75])
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"max |snr_hat(g x) - snr_hat(x)| for g in (1e-3, 10), 20 files: {max_scale_err:.2e} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
