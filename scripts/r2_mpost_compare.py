"""exp/r2-mpost: candidate vs gru_v5b_off from eval.py per-file CSVs (outputs/<name>/eval_<tag>.csv).

Prints overall / per-SNR / per-difficulty PESQ and SNR with paired deltas (delta, SE, z), deltas for the
top-5 headroom noise types of docs/r2_noise_analysis.md, the level probe and the round-2 promotion check
(after scripts/r2_v6_compare.py on exp/r2-v6).
Usage: python scripts/r2_mpost_compare.py [--cand gru_v5b_mpost] > logs/r2_mpost_compare.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired import paired_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOP5 = ["machine", "wind", "aircraft_cockpit", "broadband_noise", "alarm_siren"]
GAINS = [-30, -20, -10, 0]
VAL_GATE = 2.6962


def load(name, tag):
    return pd.read_csv(ROOT / "outputs" / name / f"eval_{tag}.csv").set_index("segment_id")


def line(lab, a, b):
    st = paired_stats(a.pesq, b.pesq.reindex(a.index))
    return (f"  {lab:24s} n={len(a):3d} ref {a.pesq.mean():.4f} cand {b.pesq.reindex(a.index).mean():.4f} "
            f"delta {st['d_mean']:+.4f} SE {st['d_se']:.4f} z {st['d_z']:+.2f} | SNR {a.snr_db.mean():.2f} -> "
            f"{b.snr_db.reindex(a.index).mean():.2f} dB"), st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="gru_v5b_off")
    ap.add_argument("--cand", default="gru_v5b_mpost")
    a = ap.parse_args()
    res = {}
    for split in ("validation", "val_remix"):
        r, c = load(a.ref, split), load(a.cand, split)
        print(f"\n===== {split}: {a.cand} vs {a.ref} (failed {int(c.pesq.isna().sum())}) =====")
        s, res[split] = line("overall", r, c)
        print(s)
        res[split]["pesq"] = c.pesq.mean()
        for col in ("target_snr_db", "noise_difficulty"):
            for k, g in r.groupby(col):
                print(line(f"{col}={k}", g, c)[0])
        for t in TOP5:
            g = r[r.noise_category == t]
            if len(g):
                print(line(f"type {t}", g, c)[0])
    print("\n===== level probe (validation, input gain dB) =====")
    probe = {n: {g: load(n, f"validation_gain{g}dB").pesq.mean() for g in GAINS} for n in (a.ref, a.cand)}
    for n in probe:
        print(f"  {n:14s} " + " ".join(f"{g:+d}:{probe[n][g]:.4f}" for g in GAINS))
    val, vr = res["validation"]["pesq"], res["val_remix"]
    c1 = round(val, 4) >= VAL_GATE
    c2 = vr["d_mean"] > 0 and vr["d_z"] >= 2
    c3 = all(probe[a.cand][g] >= probe[a.ref][g] - 0.02 for g in GAINS)
    print(f"\n===== round-2 promotion check =====\n  {a.cand}: val {val:.4f} >= {VAL_GATE} {c1} | val_remix delta "
          f"{vr['d_mean']:+.4f} z {vr['d_z']:+.2f} {c2} | probe {c3} -> {'PASS' if c1 and c2 and c3 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
