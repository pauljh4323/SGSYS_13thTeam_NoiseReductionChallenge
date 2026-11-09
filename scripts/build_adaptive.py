"""Build configs/dsp_adaptive.yaml from a per-file sweep (train subset only) and snr_hat per file.

Anchors = median snr_hat per target SNR (0/5/10/15 dB) on the same train subset. Each file is
assigned to the group of its nearest anchor. Global best = the grid point with the best mean PESQ
over all files. Per group: the best grid point is kept only if its group-mean PESQ beats the global
best's group mean by >= --min-gain; otherwise the global best's values are used for that anchor.
Validation / val_remix are never read.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import yaml

from _common import REPO_ROOT

KEYS = ("xi_min_db", "hrnr_rho", "noise_bias")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="dsp_adaptive_s1")
    ap.add_argument("--snr-hat", default="logs/snr_hat_train_sub400.csv")
    ap.add_argument("--base", default="configs/dsp_edge.yaml")
    ap.add_argument("--out", default="configs/dsp_adaptive.yaml")
    ap.add_argument("--min-gain", type=float, default=0.01)
    args = ap.parse_args(argv)

    logs = REPO_ROOT / "logs"
    summ = pd.read_csv(logs / f"sweep_{args.sweep}.csv")
    summ = summ[summ.kind == "grid"].set_index("name")
    pf = pd.read_csv(logs / f"sweep_{args.sweep}_perfile.csv")
    pf = pf[pf.name.isin(summ.index)]
    sh = pd.read_csv(REPO_ROOT / args.snr_hat)[["segment_id", "snr_hat_db"]]
    pf = pf.merge(sh, on="segment_id", how="left", validate="many_to_one")
    assert pf.snr_hat_db.notna().all(), "snr_hat missing for some sweep files"
    assert pf.groupby("name").size().eq(len(sh)).all(), "sweep/snr_hat file sets differ"

    anchors = (sh.merge(pf[["segment_id", "target_snr_db"]].drop_duplicates(), on="segment_id")
               .groupby("target_snr_db").snr_hat_db.median().sort_index())
    a = anchors.to_numpy()
    pf["group"] = np.abs(pf.snr_hat_db.to_numpy()[:, None] - a[None, :]).argmin(axis=1)

    means = pf.groupby(["group", "name"]).pesq.mean().unstack("name")  # group x point
    overall = pf.groupby("name").pesq.mean()
    g_best = overall.idxmax()
    vals = {k: [] for k in KEYS}
    lines = [f"global best {g_best}: " + ", ".join(f"{k} {summ.at[g_best, 'p_' + k]:g}" for k in KEYS)
             + f" (train mean {overall[g_best]:.4f})"]
    for gi, (tgt, anc) in enumerate(anchors.items()):
        row = means.loc[gi]
        best = row.idxmax()
        gain = row[best] - row[g_best]
        keep = gain >= args.min_gain
        pick = best if keep else g_best
        for k in KEYS:
            vals[k].append(float(summ.at[pick, "p_" + k]))
        n = int((pf.name == g_best).mul(pf.group == gi).sum())
        lines.append(f"group {gi} (target {tgt:g} dB, anchor {anc:.2f} dB, n={n}): best {best} "
                     + ", ".join(f"{k} {summ.at[best, 'p_' + k]:g}" for k in KEYS)
                     + f" {row[best]:.4f} vs global {row[g_best]:.4f} (+{gain:.4f}) -> "
                     + ("KEPT" if keep else "global"))
    report = "\n".join(lines)
    print(report)

    with open(REPO_ROOT / args.base, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    params = dict(base["params"])
    params.update({"adapt": True, "adapt_anchors_db": [round(float(x), 2) for x in a],
                   **{f"adapt_{k}": vals[k] for k in KEYS}})
    header = ("# dsp-adaptive: dsp_edge (dsp-v4) + rule-based SNR-adaptive xi_min_db / hrnr_rho / noise_bias,\n"
              f"# built by scripts/build_adaptive.py from logs/sweep_{args.sweep}{{,_perfile}}.csv and "
              f"{args.snr_hat} (400-file train subset only).\n"
              "# anchors = train median snr_hat per target SNR 0/5/10/15 dB; values linearly interpolated over\n"
              f"# snr_hat, clamped outside. Per-group best kept only if >= {args.min_gain} over the global best.\n"
              + "".join(f"# {ln}\n" for ln in lines))
    out = REPO_ROOT / args.out
    out.write_text(header + yaml.safe_dump({"method": base["method"], "params": params},
                                           sort_keys=False, default_flow_style=None),
                   encoding="utf-8")
    print(f"-> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
