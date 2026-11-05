"""Enhance a split: noisy wavs -> outputs/<config>/<split>/<same relative path>.wav (16-bit PCM)."""
from __future__ import annotations

import argparse
import sys
import time

from _common import add_common_args, load_config, output_path

from nrc.io import load_manifest, read_row, resolve_data_root, stratified_subset, write_wav_pcm16
from nrc.methods import get_method


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    root = resolve_data_root(args.data_root)
    manifest = stratified_subset(load_manifest(root, args.split), args.subset)
    method = get_method(cfg["method"], **cfg["params"])

    started = time.perf_counter()
    total_clipped = files_clipped = 0
    for _, row in manifest.iterrows():
        noisy = read_row(root, row, "noisy")
        y = method(noisy, int(row["sample_rate_hz"]))
        if y.shape != noisy.shape:
            raise ValueError(f"{row['segment_id']}: output length {y.shape} != {noisy.shape}")
        n = write_wav_pcm16(output_path(cfg["name"], args.split, row["noisy_path"]), y,
                            int(row["sample_rate_hz"]))
        total_clipped += n
        files_clipped += n > 0
    elapsed = time.perf_counter() - started
    print(f"{cfg['name']} [{args.split}]: wrote {len(manifest)} files to "
          f"outputs/{cfg['name']}/{args.split}/ in {elapsed:.1f}s; "
          f"clipped samples {total_clipped} in {files_clipped} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
