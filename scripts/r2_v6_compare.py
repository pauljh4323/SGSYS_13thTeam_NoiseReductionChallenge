"""Round-2 model comparison (exp/r2-v6, reused for exp/r2-v7): candidates vs a reference from eval.py per-file CSVs.

Reads outputs/<name>/eval_<split>.csv (validation, val_remix, validation level probes) and prints
overall / per-SNR / per-difficulty PESQ and SNR, paired deltas vs gru_v5b_off (delta, SE, z),
weak-5 category deltas, phase-headroom fraction and the round-2 promotion check.
Usage: python scripts/r2_v6_compare.py > logs/r2_v6_compare.txt   (defaults = the exp/r2-v6 run)
       python scripts/r2_v6_compare.py --cands gru_v7a_off gru_v7a_e300_off gru_v5b_s2_off            --headroom gru_v7a_off gru_v7a_e300_off > logs/r2_v7_compare.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired import paired_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEAK5 = ["aircraft_cockpit", "factory", "broadband_noise", "destroyer_operations", "hf_channel"]
HEADROOM = {"validation": 0.293, "val_remix": 0.261}  # clean phase on v5b magnitude (docs/round2_plan.md)
GAINS = [-30, -20, -10, 0]
VAL_GATE = 2.6962  # current r2-prep submission (gru_v5b_off) validation PESQ


def load(name: str, tag: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / name / f"eval_{tag}.csv").set_index("segment_id")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="gru_v5b_off")
    ap.add_argument("--cands", nargs="+", default=["gru_v6a_off", "gru_v6b_off"])
    ap.add_argument("--headroom", nargs="*", default=["gru_v6a_off"], help="models to report phase-headroom fraction for")
    ap.add_argument("--types", nargs="+", default=WEAK5,
                    help="noise categories for the per-type delta block (default weak-5; exp/r2-dpesq: top-5 headroom types)")
    a = ap.parse_args()
    REF, CANDS = a.ref, a.cands
    names = [REF] + CANDS
    res = {}
    for split in ("validation", "val_remix"):
        print(f"\n===== {split} =====")
        dfs = {n: load(n, split) for n in names}
        rows = []
        for n, d in dfs.items():
            r = {"model": n, "n": len(d), "failed": int(d.pesq.isna().sum()),
                 "pesq": d.pesq.mean(), "snr_db": d.snr_db.mean()}
            for s, g in d.groupby("target_snr_db"):
                r[f"pesq@{s:g}"] = g.pesq.mean()
            for s, g in d.groupby("noise_difficulty"):
                r[f"pesq_{s}"] = g.pesq.mean()
            if n != REF:
                st = paired_stats(dfs[REF].pesq, d.pesq.reindex(dfs[REF].index))
                r.update(delta=st["d_mean"], se=st["d_se"], z=st["d_z"])
                if n in a.headroom:
                    r["headroom_frac"] = st["d_mean"] / HEADROOM[split]
                res[(n, split)] = r
            rows.append(r)
        print(pd.DataFrame(rows).set_index("model").T.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\n[{split}] per-SNR paired deltas vs {REF} (delta / SE / z)")
        for n in CANDS:
            for s, g in dfs[REF].groupby("target_snr_db"):
                st = paired_stats(g.pesq, dfs[n].pesq.reindex(g.index))
                print(f"  {n} {s:>5g} dB n={st['d_n']:3d} {st['d_mean']:+.4f} SE {st['d_se']:.4f} z {st['d_z']:+.2f}")
        print(f"\n[{split}] categories {' '.join(a.types)}: n, {REF} PESQ, delta per candidate")
        for c in a.types:
            idx = dfs[REF].index[dfs[REF].noise_category == c]
            line = f"  {c:22s} n={len(idx):3d} v5b {dfs[REF].pesq[idx].mean():.4f}"
            for n in CANDS:
                st = paired_stats(dfs[REF].pesq[idx], dfs[n].pesq.reindex(idx))
                line += f" | {n} {st['d_mean']:+.4f} (SE {st['d_se']:.4f})"
            print(line)

    print("\n===== level probe (validation, input gain dB) =====")
    probe = {}
    for n in names:
        probe[n] = {g: load(n, f"validation_gain{g}dB").pesq.mean() for g in GAINS}
        print(f"  {n:12s} " + " ".join(f"{g:+d}:{probe[n][g]:.4f}" for g in GAINS))

    print("\n===== round-2 promotion check =====")
    for n in CANDS:
        val = res[(n, "validation")]["pesq"]
        vr = res[(n, "val_remix")]
        c1 = round(val, 4) >= VAL_GATE
        c2 = vr["delta"] > 0 and vr["z"] >= 2
        c3 = all(probe[n][g] >= probe[REF][g] - 0.02 for g in GAINS)
        print(f"  {n}: val {val:.4f} >= {VAL_GATE} {c1} | val_remix delta {vr['delta']:+.4f} z {vr['z']:+.2f} {c2} "
              f"| probe {c3} -> {'PASS' if c1 and c2 and c3 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
