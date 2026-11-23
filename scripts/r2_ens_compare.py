"""Round-2 ensemble comparison (exp/r2-ens): candidates vs the current submission from eval.py per-file CSVs.

Reads outputs/<name>/eval_<tag>.csv (validation, val_remix, validation level probes) and prints overall / per-SNR /
per-difficulty PESQ and SNR, paired deltas vs the reference (delta, SE, z; overall and per SNR), probe levels, main.py
CPU time (logs/time_r2ens_<name>.log) and the round-2 rule:
  val PESQ >= ref val, val_remix paired delta > 0 with z >= 2, every probe level >= ref probe floor.
Usage: python scripts/r2_ens_compare.py > logs/r2_ens_compare.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired import paired_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAINS = [-30, -20, -10, 0]


def load(name: str, tag: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / name / f"eval_{tag}.csv").set_index("segment_id")


def cpu_time(name: str):
    p = ROOT / "logs" / f"time_r2ens_{name}.log"
    m = re.search(r"processed=(\d+).*total=([\d.]+)s", p.read_text(encoding="utf-8")) if p.is_file() else None
    return (float(m.group(2)), int(m.group(1))) if m else (float("nan"), 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="gru_v8p2x_rmsn")
    ap.add_argument("--cands", nargs="+", default=["ens_e1", "ens_e1g", "ens_e2"])
    ap.add_argument("--val-gate", type=float, default=3.4867)
    ap.add_argument("--probe-floor", type=float, default=3.467)
    ap.add_argument("--min-z", type=float, default=2.0)
    a = ap.parse_args()
    names = [a.ref] + a.cands
    summ = {}
    for split in ("validation", "val_remix"):
        print(f"\n===== {split} =====")
        dfs = {n: load(n, split) for n in names}
        rows = []
        for n, d in dfs.items():
            r = {"model": n, "n": len(d), "failed": int(d.pesq.isna().sum()), "clipped": int(d.n_clipped.sum()),
                 "pesq": d.pesq.mean(), "snr_db": d.snr_db.mean()}
            for s, g in d.groupby("target_snr_db"):
                r[f"pesq@{s:g}"] = g.pesq.mean()
            for s, g in d.groupby("noise_difficulty"):
                r[f"pesq_{s}"] = g.pesq.mean()
            if n != a.ref:
                st = paired_stats(dfs[a.ref].pesq, d.pesq.reindex(dfs[a.ref].index))
                r.update(delta=st["d_mean"], se=st["d_se"], z=st["d_z"])
            summ[(n, split)] = r
            rows.append(r)
        print(pd.DataFrame(rows).set_index("model").T.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"\n[{split}] per-SNR paired deltas vs {a.ref} (delta / SE / z)")
        ref = dfs[a.ref]
        for n in a.cands:
            d = dfs[n].reindex(ref.index)
            for s in sorted(ref.target_snr_db.unique()):
                idx = ref.index[ref.target_snr_db == s]
                st = paired_stats(ref.pesq[idx], d.pesq[idx])
                print(f"  {n:10s} {s:4g} dB n={st['d_n']:3d} {st['d_mean']:+.4f} SE {st['d_se']:.4f} z {st['d_z']:+.2f}")
    print("\n===== level probe (validation, input gain dB) =====")
    probe = {n: {g: load(n, f"validation_gain{g}dB").pesq.mean() for g in GAINS} for n in names}
    print(pd.DataFrame(probe).T.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n===== main.py CPU time (164 validation files, same batch) =====")
    for n in names:
        t, k = cpu_time(n)
        print(f"  {n:16s} total {t:6.1f} s  ({t / max(k, 1):.3f} s/file, n={k})")
    print(f"\n===== round-2 rule vs {a.ref}: val >= {a.val_gate}, val_remix d > 0 and z >= {a.min_z}, "
          f"probe >= {a.probe_floor} =====")
    passing = []
    for n in a.cands:
        v, vr = summ[(n, "validation")], summ[(n, "val_remix")]
        c1 = round(v["pesq"], 4) >= a.val_gate
        c2 = vr["delta"] > 0 and vr["z"] >= a.min_z
        c3 = all(probe[n][g] >= a.probe_floor for g in GAINS)
        ok = c1 and c2 and c3
        if ok:
            passing.append((v["pesq"], n))
        print(f"  {n:10s} val {v['pesq']:.4f} [{'ok' if c1 else 'FAIL'}]  val_remix d {vr['delta']:+.4f} "
              f"(SE {vr['se']:.4f}, z {vr['z']:+.2f}) [{'ok' if c2 else 'FAIL'}]  probe min "
              f"{min(probe[n].values()):.4f} [{'ok' if c3 else 'FAIL'}]  -> {'PASS' if ok else 'FAIL'}")
    print(f"\nwinner: {max(passing)[1] if passing else 'none (submission unchanged)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
