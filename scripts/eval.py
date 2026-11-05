"""Evaluate a config on a split: PESQ-WB, SI-SDR, STOI, course SNR.

Enhances in memory (float64, as in the course notebook) unless --from-outputs is given,
in which case the 16-bit wavs written by scripts/enhance.py are scored.
The method only ever sees the noisy waveform; clean audio is read afterwards for scoring.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from _common import REPO_ROOT, add_common_args, git_hash, load_config, output_dir, output_path

from nrc.io import load_manifest, read_row, read_wav, resolve_data_root, stratified_subset, to_pcm16
from nrc.methods import get_method
from nrc.metrics import pesq_wb, si_sdr, snr_db, stoi

META_COLUMNS = ("segment_id", "noise_category", "noise_code", "noise_difficulty", "target_snr_db")
_METHOD_CACHE = {}


def _get_method(name, params):
    key = (name, tuple(sorted(params.items())))
    if key not in _METHOD_CACHE:
        _METHOD_CACHE[key] = get_method(name, **params)
    return _METHOD_CACHE[key]


def evaluate_row(task) -> dict:
    root, row, cfg, split, from_outputs = task
    sr = int(row["sample_rate_hz"])
    noisy = read_row(root, row, "noisy")
    t0 = time.perf_counter()
    if from_outputs:
        y = read_wav(output_path(cfg["name"], split, row["noisy_path"]), sr, row["num_samples"])
        n_clipped = 0
    else:
        y = _get_method(cfg["method"], cfg["params"])(noisy, sr)
        n_clipped = to_pcm16(y)[1]
    enhance_s = time.perf_counter() - t0
    if y.shape != noisy.shape:
        raise ValueError(f"{row['segment_id']}: output length {y.shape} != {noisy.shape}")
    clean = read_row(root, row, "clean")  # scoring only, after enhancement
    pesq_v, pesq_err = pesq_wb(clean, y, sr)
    pesq_in, _ = pesq_wb(clean, noisy, sr)
    snr_v, snr_in = snr_db(clean, y), snr_db(clean, noisy)
    return {**{c: row[c] for c in META_COLUMNS},
            "pesq": pesq_v, "pesq_error": pesq_err, "delta_pesq": pesq_v - pesq_in,
            "si_sdr": si_sdr(clean, y), "stoi": stoi(clean, y, sr),
            "snr_db": snr_v, "delta_snr_db": snr_v - snr_in,
            "n_clipped": n_clipped, "enhance_s": enhance_s}


def summarize(detail: pd.DataFrame, by=None) -> pd.DataFrame:
    groups = detail.groupby(by, sort=True) if by else detail.groupby(lambda _: "all")
    out = groups.agg(n=("segment_id", "size"), pesq_mean=("pesq", "mean"),
                     pesq_std=("pesq", "std"), pesq_valid=("pesq", "count"),
                     delta_pesq=("delta_pesq", "mean"), si_sdr=("si_sdr", "mean"),
                     stoi=("stoi", "mean"), snr_db=("snr_db", "mean"),
                     delta_snr_db=("delta_snr_db", "mean"))
    out.insert(1, "failed", out.n - out.pop("pesq_valid"))
    return out.reset_index(names=by or "group")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--from-outputs", action="store_true",
                        help="score outputs/<config>/<split>/ wavs instead of enhancing in memory")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    root = resolve_data_root(args.data_root)
    manifest = stratified_subset(load_manifest(root, args.split), args.subset)
    get_method(cfg["method"], **cfg["params"])  # fail fast on bad config
    import pesq  # noqa: F401  # fail fast if PESQ is missing; never silently omit it

    tasks = [(root, row.to_dict(), cfg, args.split, args.from_outputs)
             for _, row in manifest.iterrows()]
    started = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(evaluate_row, tasks, chunksize=4))
    else:
        rows = [evaluate_row(t) for t in tasks]
    elapsed = time.perf_counter() - started
    detail = pd.DataFrame(rows)
    if len(detail) != len(manifest):
        raise RuntimeError("Not every manifest entry was evaluated.")

    out_dir = output_dir(cfg["name"], args.split).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_sub{args.subset}" if args.subset else ""
    detail.to_csv(out_dir / f"eval_{args.split}{tag}.csv", index=False)

    tables = {"overall": summarize(detail),
              "by target_snr_db": summarize(detail, "target_snr_db"),
              "by noise_difficulty": summarize(detail, "noise_difficulty")}
    n_failed = int(detail.pesq.isna().sum())
    n_clipped = int(detail.n_clipped.sum())
    source = "16-bit outputs" if args.from_outputs else "in-memory float"
    header = (f"{cfg['name']} ({cfg['method']}) | {args.split} n={len(detail)} | {source} | "
              f"jobs={args.jobs} | {elapsed:.1f}s | PESQ failed={n_failed} | "
              f"clipped samples={n_clipped}")
    text = header + "\n" + "\n".join(
        f"\n[{name}]\n{t.to_string(index=False, float_format=lambda v: f'{v:.4f}')}"
        for name, t in tables.items())
    logs = REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    stem = f"eval_{cfg['name']}_{args.split}{tag}"
    (logs / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([t.assign(table=name).rename(columns={t.columns[0]: "key"})
               for name, t in tables.items()]).to_csv(logs / f"{stem}_summary.csv", index=False)
    print(text)
    if n_failed:
        print("PESQ failures:\n" + detail[detail.pesq.isna()][["segment_id", "pesq_error"]]
              .to_string(index=False))

    by_snr = detail.groupby("target_snr_db").pesq.mean()
    notes = (f"split={args.split} n={len(detail)} failed={n_failed} "
             f"snr={detail.snr_db.mean():.3f} stoi={detail.stoi.mean():.4f} "
             f"clipped={n_clipped} src={'outputs' if args.from_outputs else 'memory'}"
             + (f" subset={args.subset}" if args.subset else "")
             + (f" | {args.notes}" if args.notes else ""))
    record = [date.today().isoformat(), git_hash(), cfg["name"], cfg["method"],
              f"{detail.pesq.mean():.4f}",
              *[f"{by_snr.get(s, np.nan):.4f}" for s in (0.0, 5.0, 10.0, 15.0)],
              f"{detail.si_sdr.mean():.3f}", notes]
    with open(REPO_ROOT / "results" / "experiments.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
