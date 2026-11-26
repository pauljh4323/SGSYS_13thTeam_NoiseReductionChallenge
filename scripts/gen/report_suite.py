"""Tables for docs/r2_gen_suite.md from eval.py per-file CSVs (exp/r2-gen; reads results only, scores nothing).

For every model with outputs/<dir>/eval_<split>[_pred].csv: mean PESQ / SNR / failures per set, the noisy input
(pesq - delta_pesq, identical inputs for all models), per-corpus PESQ (noise_collection joined from the set's
samples.csv), per-SNR PESQ and paired PESQ deltas vs gru_v5b_off (scripts/paired.py, 95 % CI = 1.96 SE).
Run: python scripts/gen/report_suite.py [--out logs/gen_report_tables.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from paired import paired_stats  # noqa: E402

MODELS = {"dsp_v8": "dsp_v8_pred", "gru_v5b_off": "gru_v5b_off", "gru_v8p2z3d": "gru_v8p2z3d",
          "gru_v8p2z3e": "gru_v8p2z3e", "gru_v8p2z4w": "gru_v8p2z4w", "ext_dfn3": "ext_dfn3",
          "ext_mpsenet": "ext_mpsenet", "ext_cmgan": "ext_cmgan"}
SETS = {"validation": "val", "gen_unseen_test": "G_unseen_test", "gen_multi_test": "G_multi_test", "gen_low": "G_low"}
REF = "gru_v5b_off"


def manifest(split: str) -> pd.DataFrame:
    p = (REPO / "data" / "noise_reduction_challenge" / "validation" / "samples.csv" if split == "validation"
         else REPO / "data" / "gen" / split / "samples.csv")
    return pd.read_csv(p, encoding="utf-8-sig")[["segment_id", "noise_collection"]]


def load(model: str, split: str) -> pd.DataFrame | None:
    d = REPO / "outputs" / MODELS[model]
    for name in (f"eval_{split}.csv", f"eval_{split}_pred.csv"):
        if (d / name).is_file():
            df = pd.read_csv(d / name)
            return df.merge(manifest(split), on="segment_id", how="left")
    return None


def fmt(x, nd=3):
    return "–" if pd.isna(x) else f"{x:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "logs" / "gen_report_tables.md"))
    args = ap.parse_args(argv)
    data = {(m, s): df for m in MODELS for s in SETS if (df := load(m, s)) is not None}
    models = [m for m in MODELS if any((m, s) in data for s in SETS)]
    L = []
    # noisy input
    L.append("### Overall PESQ / SNR (dB) / PESQ failures\n")
    L.append("| model | " + " | ".join(SETS.values()) + " |")
    L.append("|---" * (len(SETS) + 1) + "|")
    noisy = {}
    for s in SETS:
        df = next((data[(m, s)] for m in models if (m, s) in data), None)
        if df is not None:
            noisy[s] = (df.pesq - df.delta_pesq, df.snr_db - df.delta_snr_db)
    L.append("| noisy input | " + " | ".join(
        f"{fmt(noisy[s][0].mean())} / {fmt(noisy[s][1].mean(), 2)}" if s in noisy else "–" for s in SETS) + " |")
    for m in models:
        cells = []
        for s in SETS:
            df = data.get((m, s))
            if df is None:
                cells.append("pending")
                continue
            fail = int(df.pesq.isna().sum())
            cells.append(f"{fmt(df.pesq.mean())} / {fmt(df.snr_db.mean(), 2)}" + (f" / {fail}F" if fail else ""))
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    # paired deltas vs REF
    L.append(f"\n### Paired PESQ delta vs {REF} (mean ± 95 % CI, z)\n")
    L.append("| model | " + " | ".join(SETS.values()) + " |")
    L.append("|---" * (len(SETS) + 1) + "|")
    for m in models:
        if m == REF:
            continue
        cells = []
        for s in SETS:
            a, b = data.get((REF, s)), data.get((m, s))
            if a is None or b is None:
                cells.append("–")
                continue
            st = paired_stats(a.set_index("segment_id").pesq, b.set_index("segment_id").pesq)
            cells.append(f"{st['d_mean']:+.3f} ± {1.96 * st['d_se']:.3f} (z {st['d_z']:.1f})")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    # per corpus
    for s in ("gen_unseen_test", "gen_multi_test", "gen_low", "validation"):
        cols = sorted({c for m in models if (m, s) in data for c in data[(m, s)].noise_collection.dropna().unique()})
        if not cols:
            continue
        L.append(f"\n### {SETS[s]}: PESQ per noise corpus (n)\n")
        n = data[next(m for m in models if (m, s) in data), s].groupby("noise_collection").size()
        L.append("| model | " + " | ".join(f"{c} ({n.get(c, 0)})" for c in cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        base = next(data[(m, s)] for m in models if (m, s) in data)
        g = (base.pesq - base.delta_pesq).groupby(base.noise_collection).mean()
        L.append("| noisy input | " + " | ".join(fmt(g.get(c)) for c in cols) + " |")
        for m in models:
            if (m, s) in data:
                g = data[(m, s)].groupby("noise_collection").pesq.mean()
                L.append(f"| {m} | " + " | ".join(fmt(g.get(c)) for c in cols) + " |")
    # per SNR
    for s in SETS:
        snrs = sorted({x for m in models if (m, s) in data for x in data[(m, s)].target_snr_db.unique()})
        if len(snrs) < 2:
            continue
        L.append(f"\n### {SETS[s]}: PESQ per input SNR\n")
        L.append("| model | " + " | ".join(f"{x:g} dB" for x in snrs) + " |")
        L.append("|---" * (len(snrs) + 1) + "|")
        for m in models:
            if (m, s) in data:
                g = data[(m, s)].groupby("target_snr_db").pesq.mean()
                L.append(f"| {m} | " + " | ".join(fmt(g.get(x)) for x in snrs) + " |")
    # pretrained meta
    L.append("\n### Pretrained inference meta (per set: rescaled files to peak 0.99, CPU s/file, threads)\n")
    for m in ("ext_dfn3", "ext_mpsenet", "ext_cmgan"):
        parts = []
        for s in SETS:
            p = REPO / "outputs" / m / f"{s}_meta.json"
            if p.is_file():
                j = json.loads(p.read_text())
                parts.append(f"{SETS[s]}: rescaled {j.get('rescaled_files', 'n/a')}, would-clip samples "
                             f"{j['clipped_samples']}, {j['cpu_s_per_file_mean']:.2f} s, params {j['params']}")
        L.append(f"- {m}: " + ("; ".join(parts) if parts else "pending"))
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
