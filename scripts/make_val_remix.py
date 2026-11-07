"""Build the secondary held-out set data/val_remix/ from validation pairs only (deterministic).

The validation split has 164 pairs, so each clean utterance i is used --per-clean (default 2) times,
giving 328 mixtures: each with noise_j = noisy_j - clean_j from a DIFFERENT validation segment j
(per-copy random derangements; the copies of i use distinct noise sources), SNR drawn uniformly
from {0, 5, 10, 15} dB, mix = clean_i + noise_j * sqrt(Pc / (Pn * 10^(snr/10))). No overall gain
change (clean is written as-is), except a joint peak guard (clean and mix scaled together) if
|mix| > --peak, to avoid 16-bit clipping.
Writes data/val_remix/{clean,noisy}/<segment_id>_vr<k>[_noisy].wav (16-bit PCM) and samples.csv
with the official manifest columns: segment_id = <clean id>_vr<k>, split = "val_remix",
target_snr_db = the drawn mixing SNR, noise_category/collection/code/difficulty from the noise
source; extra columns noise_source_id and noise_src_snr_db (the noise source's original target
SNR). Paths are relative to data/val_remix. Train audio is never read.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import soundfile as sf

from _common import REPO_ROOT  # noqa: F401  (sys.path setup)

from nrc.io import EXTRA_SPLITS, load_manifest, read_row, resolve_data_root, to_pcm16

SNRS = (0.0, 5.0, 10.0, 15.0)
NOISE_META = ("noise_category", "noise_collection", "noise_code", "noise_difficulty")


def noise_sources(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """(k, n): each row is a derangement of range(n); the k sources of each column are distinct."""
    rows: list[np.ndarray] = []
    while len(rows) < k:
        perm = rng.permutation(n)
        if (perm == np.arange(n)).any() or any((perm == r).any() for r in rows):
            continue
        rows.append(perm)
    return np.stack(rows)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    return np.resize(x, n) if x.size != n else x  # np.resize tiles or crops


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--per-clean", type=int, default=2, help="mixtures per clean utterance")
    parser.add_argument("--peak", type=float, default=0.99)
    args = parser.parse_args(argv)

    root = resolve_data_root(args.data_root)
    df = load_manifest(root, "validation").sort_values("segment_id").reset_index(drop=True)
    out = EXTRA_SPLITS["val_remix"]
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "noisy").mkdir(parents=True, exist_ok=True)
    clean = [read_row(root, r, "clean") for _, r in df.iterrows()]
    noise = [read_row(root, r, "noisy") - c for (_, r), c in zip(df.iterrows(), clean)]

    rng = np.random.default_rng(args.seed)
    n_clean, k = len(df), args.per_clean
    src = noise_sources(rng, n_clean, k)
    snrs = rng.choice(np.array(SNRS), size=(k, n_clean))
    rows, n_guard = [], 0
    for copy in range(k):
        for i, r in df.iterrows():
            j, snr = int(src[copy, i]), float(snrs[copy, i])
            c, n = clean[i], fit_length(noise[j], clean[i].size)
            pc, pn = float(np.mean(c ** 2)), float(np.mean(n ** 2))
            scale = np.sqrt(pc / (pn * 10 ** (snr / 10))) if pc > 0 and pn > 0 else 0.0
            mix = c + n * scale
            pk = float(np.max(np.abs(mix)))
            if pk > args.peak:  # only to avoid 16-bit clipping
                c, mix, n_guard = c * (args.peak / pk), mix * (args.peak / pk), n_guard + 1
            sid = f"{r.segment_id}_vr{copy}"
            crel, nrel = f"clean/{sid}.wav", f"noisy/{sid}_noisy.wav"
            for rel, x in ((crel, c), (nrel, mix)):
                pcm, clipped = to_pcm16(x)
                assert clipped == 0, f"{rel}: {clipped} clipped samples"
                sf.write(str(out / rel), pcm, int(r.sample_rate_hz), subtype="PCM_16")
            nr = df.iloc[j]
            rows.append({"segment_id": sid, "clean_path": crel, "noisy_path": nrel,
                         "sample_rate_hz": int(r.sample_rate_hz), "num_samples": int(c.size),
                         **{key: nr[key] for key in NOISE_META}, "target_snr_db": snr,
                         "split": "val_remix", "noise_source_id": nr.segment_id,
                         "noise_src_snr_db": float(nr.target_snr_db)})
    man = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)
    assert (man.noise_source_id != man.segment_id.str.rsplit("_vr", n=1).str[0]).all()
    man.to_csv(out / "samples.csv", index=False, encoding="utf-8")
    load_manifest(root, "val_remix")  # validate what we wrote
    print(f"val_remix: {len(man)} mixtures -> {out} (seed {args.seed}, peak-guarded {n_guard})")
    print("per SNR:", man.target_snr_db.value_counts().sort_index().to_dict())
    print("per difficulty:", man.noise_difficulty.value_counts().sort_index().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
