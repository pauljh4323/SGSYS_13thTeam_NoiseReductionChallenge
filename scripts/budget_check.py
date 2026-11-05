"""Budget check for gru_mask configs with random-init weights (no training needed).

Per --config: param count (limit < 5M), inference checkpoint size, and CPU inference time over all
validation noisy files (per-file forward, threads = min(4, cpu) as in nrc.methods.gru_mask; file
reading timed separately). Exits 1 with a loud message if any config exceeds a budget.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from _common import load_config

from nrc.io import load_manifest, read_row, resolve_data_root

PARAM_LIMIT = 5_000_000
TIME_LIMIT_S = 180.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args(argv)

    import torch
    from nrc.models.gru_mask import GRUMask, count_params, save_weights

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    root = resolve_data_root(args.data_root)
    df = load_manifest(root, args.split)
    t0 = time.perf_counter()
    waves = [read_row(root, r, "noisy").astype(np.float32) for _, r in df.iterrows()]
    read_s = time.perf_counter() - t0
    print(f"{args.split}: {len(waves)} files, read {read_s:.1f}s, torch threads {torch.get_num_threads()}")

    failures = []
    for cfg_path in args.config:
        cfg = load_config(cfg_path)
        torch.manual_seed(0)
        model = GRUMask(**cfg.get("model", {})).eval()
        n_params = count_params(model)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "w.pt"
            save_weights(p, model)
            size_mb = p.stat().st_size / 1e6
        t0 = time.perf_counter()
        with torch.inference_mode():
            for x in waves:
                y = model(torch.from_numpy(x)[None])[0]
                assert y.shape[-1] == x.size
        infer_s = time.perf_counter() - t0
        ok_p, ok_t = n_params < PARAM_LIMIT, infer_s < TIME_LIMIT_S
        print(f"{cfg['name']}: params {n_params:,} ({'OK' if ok_p else 'OVER'} < {PARAM_LIMIT:,}), "
              f"ckpt {size_mb:.2f} MB, CPU infer {len(waves)} files {infer_s:.1f}s "
              f"({'OK' if ok_t else 'OVER'} < {TIME_LIMIT_S:.0f}s)")
        if not ok_p:
            failures.append(f"{cfg['name']}: {n_params:,} params >= {PARAM_LIMIT:,}")
        if not ok_t:
            failures.append(f"{cfg['name']}: inference {infer_s:.1f}s >= {TIME_LIMIT_S:.0f}s")
    if failures:
        print("!!! BUDGET EXCEEDED !!!\n" + "\n".join("  " + f for f in failures))
        return 1
    print("budget check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
