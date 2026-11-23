"""N=1 identity check: gru_ens with one member == gru_mask level_mode rms_norm (bit-identical float output).

python scripts/ens_identity.py [--n 20] [--weights weights/gru_v8p2x.pt]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import add_common_args  # noqa: F401  (sets sys.path to src/)
from nrc.io import load_manifest, read_row, resolve_data_root, split_root
from nrc.methods import get_method


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--weights", default="weights/gru_v8p2x.pt")
    ap.add_argument("--target-rms", type=float, default=0.079130)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)
    root = resolve_data_root(args.data_root)
    man = load_manifest(root, "validation").head(args.n)
    single = get_method("gru_mask", weights=args.weights, device="cpu", level_norm=True,
                        level_mode="rms_norm", target_rms=args.target_rms)
    ens = get_method("gru_ens", weights=[args.weights], combine="mean", device="cpu",
                     target_rms=args.target_rms)
    vroot = split_root(root, "validation")
    n_same = 0
    for _, row in man.iterrows():
        x = read_row(vroot, row.to_dict(), "noisy")
        a, b = single(x, 16000), ens(x, 16000)
        same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        n_same += same
        print(f"{row['segment_id']}: {'identical' if same else f'DIFF max {np.max(np.abs(a - b)):.3e}'}")
    print(f"N=1 identity: {n_same}/{len(man)} bit-identical ({args.weights})")
    return 0 if n_same == len(man) else 1


if __name__ == "__main__":
    sys.exit(main())
