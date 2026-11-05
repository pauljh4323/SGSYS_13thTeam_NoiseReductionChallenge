"""Input level statistics (dBFS) of the noisy files of train and validation.

Two measures per file (same definitions as nrc.methods.gru_mask.measure_level_dbfs):
  rms     whole-file RMS
  active  RMS of the loudest 50% of non-overlapping 32 ms (512-sample) frames
Prints median / p5 / p95 per split and writes logs/level_stats.csv (one row per file).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from _common import REPO_ROOT

from nrc.io import load_manifest, read_row, resolve_data_root
from nrc.methods.gru_mask import measure_level_dbfs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)
    root = resolve_data_root(args.data_root)
    rows = []
    for split in ("train", "validation"):
        for _, row in load_manifest(root, split).iterrows():
            x = read_row(root, row, "noisy")
            rows.append({"split": split, "segment_id": row["segment_id"],
                         "target_snr_db": row["target_snr_db"],
                         "rms_dbfs": measure_level_dbfs(x, "rms"),
                         "active_dbfs": measure_level_dbfs(x, "active")})
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "logs" / "level_stats.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{'split':<11}{'measure':<8}{'n':>6}{'median':>9}{'p5':>9}{'p95':>9}")
    for split, g in df.groupby("split", sort=False):
        for m in ("rms", "active"):
            v = g[f"{m}_dbfs"].to_numpy()
            v = v[np.isfinite(v)]
            print(f"{split:<11}{m:<8}{v.size:>6}{np.median(v):>9.2f}"
                  f"{np.percentile(v, 5):>9.2f}{np.percentile(v, 95):>9.2f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
