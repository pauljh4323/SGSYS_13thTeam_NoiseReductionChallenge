"""exp/r2-v6 sanity: warm-started gru_v6a (crm) / gru_v6b (noise_aware) models reproduce gru_v5b exactly at init,
plus CPU timings of the noise-aware features (per file and per batch of 16, 1..N worker processes) and of
inference per file for v5b / v6a / v6b.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import REPO_ROOT, load_config

from nrc.features import batch_features, noise_aware_features
from nrc.io import load_manifest, read_row, resolve_data_root, split_root, stratified_subset


def build(cfg_name, ck):
    from nrc.models.gru_mask import GRUMask, load_partial
    cfg = load_config(REPO_ROOT / "configs" / "dl" / f"{cfg_name}.yaml")
    m = GRUMask(**cfg["model"])
    left = load_partial(m, ck)
    return m.eval(), left


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args(argv)
    import torch
    from nrc.models.gru_mask import load_weights
    torch.set_num_threads(4)
    root0 = resolve_data_root(a.data_root)
    man = stratified_subset(load_manifest(root0, "validation"), a.n)
    root = split_root(root0, "validation")
    xs = np.stack([read_row(root, r, "noisy").astype(np.float32) for _, r in man.iterrows()])
    ck = torch.load(REPO_ROOT / "weights" / "gru_v5b.pt", map_location="cpu", weights_only=True)
    base = load_weights(REPO_ROOT / "weights" / "gru_v5b.pt")
    ok = True
    times = {}
    with torch.inference_mode():
        ref = [base(torch.from_numpy(x)[None])[0] for x in xs]
        t0 = time.perf_counter()
        for x in xs:
            base(torch.from_numpy(x)[None])
        times["gru_v5b"] = (time.perf_counter() - t0) / len(xs)
        for name in ("gru_v6a", "gru_v6b"):
            m, left = build(name, ck)
            t0 = time.perf_counter()
            outs = [m(torch.from_numpy(x)[None])[0] for x in xs]
            times[name] = (time.perf_counter() - t0) / len(xs)
            same = all(torch.equal(o, r) for o, r in zip(outs, ref))
            ok &= same
            print(f"{name}: keys left at init {left}; output == gru_v5b on {len(xs)} files: {same}")
    print("inference s/file (3 s clip, CPU, 4 torch threads): " + ", ".join(f"{k} {v:.3f}" for k, v in times.items()))
    f = noise_aware_features(xs[0])
    print(f"features shape {f.shape}, finite {np.isfinite(f).all()}")
    t0 = time.perf_counter()
    for x in xs:
        noise_aware_features(x)
    per_file = (time.perf_counter() - t0) / len(xs)
    print(f"noise-aware features: {per_file * 1000:.1f} ms/file single process -> {16 * per_file:.3f} s per batch of 16")
    batches = [xs[np.arange(i, i + 16) % len(xs)] for i in range(0, 16 * 8, 16)]  # 8 batches of 16
    for w in a.workers:
        with ProcessPoolExecutor(max_workers=w) as pool:
            list(pool.map(batch_features, batches[:w]))  # warm-up imports
            t0 = time.perf_counter()
            list(pool.map(batch_features, batches))
            dt = (time.perf_counter() - t0) / len(batches)
        print(f"loader throughput with {w} workers: {dt:.3f} s per batch of 16")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
