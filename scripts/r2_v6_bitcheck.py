"""Bit-identity check of gru_mask inference across code changes (branch exp/r2-v6).

--save REF.npz     enhance the first N validation files (stratified subset, seed 0) with CONFIG and save the outputs
--compare REF.npz  enhance the same files again and report max |diff| (must be exactly 0)
Also prints the inference time per file.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from _common import REPO_ROOT, load_config

from nrc.io import load_manifest, read_row, resolve_data_root, split_root, stratified_subset
from nrc.methods import get_method


def run(config, n, data_root):
    cfg = load_config(config)
    man = stratified_subset(load_manifest(data_root, "validation"), n)
    root = split_root(data_root, "validation")
    fn = get_method(cfg["method"], **cfg["params"])
    out, times = {}, []
    fn(read_row(root, man.iloc[0], "noisy"), 16000)  # warm-up (model load)
    for _, r in man.iterrows():
        x = read_row(root, r, "noisy")
        t0 = time.perf_counter()
        out[r["segment_id"]] = fn(x, 16000)
        times.append(time.perf_counter() - t0)
    print(f"{cfg['name']}: {len(out)} files, inference {np.mean(times):.3f} s/file "
          f"(median {np.median(times):.3f}, 3 s clips, CPU)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "dl" / "gru_v5b_off.yaml"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args(argv)
    out = run(a.config, a.n, resolve_data_root(a.data_root))
    if a.save:
        np.savez(a.save, **out)
        print(f"saved {a.save}")
    if a.compare:
        ref = np.load(a.compare)
        diffs = [float(np.max(np.abs(ref[k] - v))) for k, v in out.items()]
        same = all(np.array_equal(ref[k], v) for k, v in out.items()) and set(ref.files) == set(out)
        print(f"max |diff| = {max(diffs):.3e}; bit-identical: {same}")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
