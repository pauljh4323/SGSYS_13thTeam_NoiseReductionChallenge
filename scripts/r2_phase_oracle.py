"""Phase headroom of gru_v5b (analysis only, ORACLE: uses the clean phase; branch exp/r2-v6).

For each validation / val_remix file: gru_v5b mask M on the noisy STFT (model grid 512/128), then
  plain  : ISTFT(M * Y)                         (= gru_v5b inference)
  oracle : ISTFT(|M * Y| * exp(j angle(S_clean))) (enhanced magnitude, clean phase)
both written as 16-bit wavs and scored with scripts/eval.py's evaluate_row (pred path). The clean signal is read
only to build the oracle bound; this is not a method. Scores go to outputs/r2_phase/ and logs/ only (not to
results/experiments.csv).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT, load_config
from r2_diag import _score_task, md

from nrc.io import load_manifest, read_row, resolve_data_root, split_root, write_wav_pcm16

OUT = REPO_ROOT / "outputs" / "r2_phase"
SPLITS = ("validation", "val_remix")
WEAK = ("aircraft_cockpit", "factory", "broadband_noise", "destroyer_operations", "hf_channel")


def enhance_all(data_root):
    import torch
    from nrc.models.gru_mask import load_weights
    torch.set_num_threads(4)
    cfg = load_config(REPO_ROOT / "configs" / "dl" / "gru_v5b_off.yaml")
    model = load_weights(REPO_ROOT / cfg["params"]["weights"], "cpu")
    for split in SPLITS:
        man = load_manifest(data_root, split)
        root = split_root(data_root, split)
        for _, r in man.iterrows():
            name = Path(r["noisy_path"]).name
            p_plain, p_orc = OUT / split / "plain" / name, OUT / split / "oracle" / name
            if p_plain.exists() and p_orc.exists():
                continue
            x = torch.from_numpy(read_row(root, r, "noisy").astype(np.float32))[None]
            s = torch.from_numpy(read_row(root, r, "clean").astype(np.float32))[None]  # oracle only
            with torch.inference_mode():
                y = model(x)[0].double().numpy()
                spec = model.stft(x)
                est = model.mask(spec.abs()) * spec
                ph = torch.angle(model.stft(s))
                yo = model.istft(torch.polar(est.abs(), ph), length=x.shape[-1])[0].double().numpy()
            write_wav_pcm16(p_plain, y)
            write_wav_pcm16(p_orc, yo)
        print(f"enhanced {split}: {len(man)}", flush=True)


def score_all(data_root, jobs):
    cfg = load_config(REPO_ROOT / "configs" / "dl" / "gru_v5b_off.yaml")
    for split in SPLITS:
        man = load_manifest(data_root, split)
        root = split_root(data_root, split)
        for kind in ("plain", "oracle"):
            csv = OUT / f"score_{split}_{kind}.csv"
            if csv.exists():
                continue
            tasks = [(root, r.to_dict(), cfg, split, "pred", OUT / split / kind / Path(r["noisy_path"]).name, 0.0, 0.0)
                     for _, r in man.iterrows()]
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                df = pd.DataFrame(list(pool.map(_score_task, tasks, chunksize=4)))
            df.to_csv(csv, index=False)
            print(f"{split} {kind}: PESQ {df.pesq.mean():.4f} SNR {df.snr_db.mean():.3f}", flush=True)


def report():
    out = ["# gru_v5b phase headroom (oracle clean phase)\n"]
    for split in SPLITS:
        p = pd.read_csv(OUT / f"score_{split}_plain.csv").set_index("segment_id")
        o = pd.read_csv(OUT / f"score_{split}_oracle.csv").set_index("segment_id")
        d = p[["noise_category", "noise_difficulty", "target_snr_db"]].copy()
        d["gru_v5b"], d["oracle_phase"] = p.pesq, o.pesq
        d["snr_gru"], d["snr_oracle"] = p.snr_db, o.snr_db
        d["delta"] = d.oracle_phase - d.gru_v5b

        def tab(g, key):
            t = g.agg(n=("delta", "size"), gru_v5b=("gru_v5b", "mean"), oracle_phase=("oracle_phase", "mean"),
                      delta=("delta", "mean"), snr_gru=("snr_gru", "mean"), snr_oracle=("snr_oracle", "mean"))
            return t.reset_index(names=key)
        rows = [tab(d.groupby(lambda _: "all"), "group"),
                tab(d.groupby("target_snr_db"), "group"),
                tab(d.groupby("noise_difficulty"), "group"),
                tab(d[d.noise_category.isin(WEAK)].groupby("noise_category"), "group"),
                tab(d[d.noise_category.isin(WEAK)].groupby(lambda _: "weak 5 (pooled)"), "group")]
        t = pd.concat(rows)
        t["group"] = t.group.astype(str)
        out.append(f"\n## {split}\n\n" + md(t))
    text = "\n".join(out) + "\n"
    (REPO_ROOT / "logs" / "r2_phase_oracle.md").write_text(text, encoding="utf-8")
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args(argv)
    root = resolve_data_root(a.data_root)
    enhance_all(root)
    score_all(root, a.jobs)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
