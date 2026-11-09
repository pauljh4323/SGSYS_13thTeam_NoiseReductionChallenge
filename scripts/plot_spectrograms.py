"""Clean / noisy / enhanced spectrograms for 4 fixed validation files (dev tool, matplotlib).

File choice (deterministic): for each target SNR 0/5/10/15 dB, the first segment_id in sorted order
with noise_difficulty == "hard" (any difficulty if none is hard). The enhanced signal is produced like
main.py (method from the config, 16-bit quantization). PESQ in the title comes from
outputs/<config>/eval_validation.csv (scripts/eval.py); this script computes no metrics.
Figures -> docs/figures/<config>_<segment_id>.png; noisy + enhanced wavs -> outputs/listen/<config>/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from _common import REPO_ROOT, load_config  # noqa: E402

from nrc.io import (load_manifest, read_row, resolve_data_root, split_root,  # noqa: E402
                    to_pcm16, write_wav_pcm16)
from nrc.methods import get_method  # noqa: E402
from nrc.stft import stft  # noqa: E402

SNRS = (0, 5, 10, 15)
MAX_BYTES = 300_000
DB_RANGE = 80.0  # colour range below the per-figure maximum


def pick_rows(manifest: pd.DataFrame) -> list[pd.Series]:
    rows = []
    for snr in SNRS:
        g = manifest[manifest.target_snr_db == snr].sort_values("segment_id")
        if g.empty:
            raise RuntimeError(f"no validation file at {snr} dB")
        hard = g[g.noise_difficulty == "hard"]
        rows.append((hard if not hard.empty else g).iloc[0])
    return rows


def spec_db(x: np.ndarray, sr: int):
    f, t, z = stft(x, sr=sr, n_fft=512, hop=128, window="hann")
    return f / 1000.0, t, 20 * np.log10(np.abs(z) + 1e-10)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    root = resolve_data_root(args.data_root)
    manifest = load_manifest(root, "validation")
    root = split_root(root, "validation")
    eval_csv = REPO_ROOT / "outputs" / cfg["name"] / "eval_validation.csv"
    if not eval_csv.is_file():
        raise FileNotFoundError(f"{eval_csv} missing: run scripts/eval.py --config {args.config} first")
    scores = pd.read_csv(eval_csv).set_index("segment_id")
    enhance = get_method(cfg["method"], **cfg["params"])
    fig_dir, listen = REPO_ROOT / "docs" / "figures", REPO_ROOT / "outputs" / "listen" / cfg["name"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    listen.mkdir(parents=True, exist_ok=True)

    for row in pick_rows(manifest):
        sr = int(row.sample_rate_hz)
        clean, noisy = read_row(root, row, "clean"), read_row(root, row, "noisy")
        enhanced = to_pcm16(enhance(noisy, sr=sr))[0].astype(np.float64) / 32768.0
        s = scores.loc[row.segment_id]
        p_out, p_in = float(s.pesq), float(s.pesq - s.delta_pesq)
        specs = [spec_db(x, sr) for x in (clean, noisy, enhanced)]
        vmax = max(float(d.max()) for _, _, d in specs)
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.6), sharex=True, constrained_layout=True)
        for ax, (f, t, d), label in zip(axes, specs, ("clean", "noisy", f"enhanced ({cfg['name']})")):
            im = ax.pcolormesh(t, f, d, vmin=vmax - DB_RANGE, vmax=vmax, cmap="magma",
                               shading="auto", rasterized=True)
            ax.set_ylabel("kHz")
            ax.set_title(label, fontsize=9, loc="left")
        axes[-1].set_xlabel("time (s)")
        fig.colorbar(im, ax=axes, label="dB", shrink=0.9)
        fig.suptitle(f"{row.segment_id} | {row.noise_category} ({row.noise_difficulty}) | "
                     f"{row.target_snr_db:g} dB SNR | PESQ {p_in:.3f} -> {p_out:.3f}", fontsize=9)
        out = fig_dir / f"{cfg['name']}_{row.segment_id}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        # 256-colour palette PNG (Pillow ships with matplotlib) keeps each figure < 300 KB
        Image.open(out).convert("RGB").quantize(colors=256).save(out, optimize=True)
        size = out.stat().st_size
        if size >= MAX_BYTES:
            raise RuntimeError(f"{out.name} is {size} bytes (limit {MAX_BYTES})")
        shutil.copyfile(root / row.noisy_path, listen / f"{row.segment_id}_noisy.wav")
        write_wav_pcm16(listen / f"{row.segment_id}_enhanced.wav", enhanced, sr)
        print(f"{out.relative_to(REPO_ROOT).as_posix()} ({size // 1024} KB) "
              f"{row.noise_category} {row.target_snr_db:g} dB PESQ {p_in:.3f} -> {p_out:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
