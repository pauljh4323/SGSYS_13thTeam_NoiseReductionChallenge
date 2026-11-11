"""Paired per-file PESQ comparison (B - A on the same files).

Library: paired_stats(a, b) with two Series indexed by segment_id.
CLI:     python scripts/paired.py A_detail.csv B_detail.csv   (eval.py per-file CSVs, column pesq)
Keep rule used by the staged sweeps: delta >= 0.002 AND delta >= 2 * SE.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

KEEP_MIN_DELTA = 0.002
KEEP_MIN_Z = 2.0


def paired_stats(a: pd.Series, b: pd.Series) -> dict:
    """Mean of b - a over files valid in both, SE = std(ddof=1)/sqrt(n), z = mean/SE."""
    d = (b - a).dropna()
    n = len(d)
    mean = float(d.mean()) if n else np.nan
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    z = mean / se if se and se > 0 else (0.0 if mean == 0 else np.inf * np.sign(mean))
    return {"d_mean": mean, "d_se": se, "d_z": z, "d_n": n}


def keeps(stats: dict, min_delta=KEEP_MIN_DELTA, min_z=KEEP_MIN_Z) -> bool:
    m, se = stats["d_mean"], stats["d_se"]
    return bool(np.isfinite(m) and m >= min_delta and (not np.isfinite(se) or m >= min_z * se))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", help="reference per-file CSV")
    ap.add_argument("b", help="candidate per-file CSV")
    args = ap.parse_args(argv)
    a = pd.read_csv(args.a).set_index("segment_id").pesq
    b = pd.read_csv(args.b).set_index("segment_id").pesq
    if set(a.index) != set(b.index):
        raise SystemExit("file sets differ")
    s = paired_stats(a, b)
    print(f"n={s['d_n']} mean A={a.mean():.4f} B={b.mean():.4f} | delta={s['d_mean']:+.4f} "
          f"SE={s['d_se']:.4f} delta/SE={s['d_z']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
