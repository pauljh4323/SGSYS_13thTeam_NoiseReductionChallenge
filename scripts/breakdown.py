"""Paired per-file PESQ breakdown (B - A) by target SNR, difficulty and noise category.

Sources:
  --sweep NAME --a POINT_A --b POINT_B   points of logs/sweep_NAME.csv by their `name` column (grid points
                                          dsp_x_000..., or the reference/stage-reference config name)
  --csv A.csv B.csv                       eval.py per-file CSVs (column pesq, indexed by segment_id)
Metadata comes from the manifest of --split. Categories: all, or only --cats. Writes to --out if given.
Analysis only; does not append to results/experiments.csv.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from _common import REPO_ROOT

from nrc.io import load_manifest, resolve_data_root
from paired import paired_stats


def sweep_series(name, point):
    s = pd.read_csv(REPO_ROOT / "logs" / f"sweep_{name}.csv")
    key = s.loc[s["name"] == point, "key"]
    if len(key) != 1:
        raise SystemExit(f"{point}: {len(key)} rows in sweep_{name}.csv")
    pf = pd.read_csv(REPO_ROOT / "logs" / f"sweep_{name}_perfile.csv")
    pf = pf[pf.key == key.iloc[0]].drop_duplicates("segment_id", keep="last")
    return pf.set_index("segment_id").pesq


def table(d, by, meta):
    rows = []
    for g, idx in meta.groupby(by).groups.items():
        sub = d.reindex(idx).dropna()
        st = paired_stats(sub["a"], sub["b"])
        rows.append({by: g, "n": st["d_n"], "A": sub["a"].mean(), "B": sub["b"].mean(),
                     "d": st["d_mean"], "SE": st["d_se"], "z": st["d_z"]})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--csv", nargs=2)
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.csv:
        a, b = (pd.read_csv(p).set_index("segment_id").pesq for p in args.csv)
        label = f"{args.csv[1]} - {args.csv[0]}"
    else:
        a, b = sweep_series(args.sweep, args.a), sweep_series(args.sweep, args.b)
        label = f"sweep {args.sweep}: {args.b} - {args.a}"
    meta = load_manifest(resolve_data_root(args.data_root), args.split).set_index("segment_id")
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    meta = meta.loc[meta.index.intersection(d.index)]
    st = paired_stats(d["a"], d["b"])
    fmt = {"index": False, "float_format": lambda v: f"{v:+.4f}"}
    lines = [f"[{label}] {args.split} n={st['d_n']} | A {d['a'].mean():.4f} B {d['b'].mean():.4f} | "
             f"d={st['d_mean']:+.4f} +- {st['d_se']:.4f} (z {st['d_z']:.2f})"]
    for by in ("target_snr_db", "noise_difficulty"):
        lines.append(f"\n[by {by}]\n" + table(d, by, meta).to_string(**fmt))
    cat = table(d, "noise_category", meta)
    if args.cats:
        cat = cat[cat.noise_category.isin(args.cats)]
    lines.append("\n[by noise_category]\n" + cat.sort_values("d", ascending=False).to_string(**fmt))
    text = "\n".join(lines)
    if args.out:
        (REPO_ROOT / args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
