"""exp/r2-mpost: SNR-adaptive mask-strength post-process for gru_mask (tuning tool, no training).

Stages (run one at a time; long ones detached, logs in logs/):
  identity  20 validation files: mask_post off == r2-prep code (bit-exact), mask_post identity block == off,
            16-bit PESQ == the recorded eval.py rows of the reference config.
  cache     run the model once per file of the train subset; save its STFT X and mask M (logs/r2_mpost_cache/).
  grid      gamma_lo x gamma_hi x (a, b) x floor x hpf on the cached M, X (16-bit PESQ, as main.py writes).
  frame     frame-level SNR_est variant of one grid point (default: the grid's best) on the same files.
  report    top-5 table with paired deltas vs identity, frame vs file, train-subset vs validation gap.
The post-process code is the method's own (nrc.methods.gru_mask: mask_snr_db, gamma_of_snr, post_mask,
synth, highpass), so a grid point reproduces enhance(..., mask_post=...) exactly.
Usage: python scripts/r2_mpost.py --stage grid --jobs 4 > logs/r2_mpost_grid.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT, load_config
from paired import paired_stats

from nrc.io import load_manifest, read_row, resolve_data_root, split_root, stratified_subset, to_pcm16
from nrc.methods import get_method
from nrc.methods import gru_mask as gm
from nrc.metrics import pesq_wb, snr_db

CACHE = REPO_ROOT / "logs" / "r2_mpost_cache"
GRID = {"gamma_lo": [0.6, 0.7, 0.8, 0.9, 1.0], "gamma_hi": [1.0, 1.1, 1.25, 1.5],
        "ab": [(0, 10), (3, 12), (5, 15)], "floor": [0.0, 0.03, 0.1], "highpass_hz": [0.0, 100.0]}
VAL_PESQ = 2.6962  # gru_v5b_off validation (results/experiments.csv, r2-prep)


def grid_points() -> pd.DataFrame:
    rows = [dict(gamma_lo=gl, gamma_hi=gh, snr_lo=float(a), snr_hi=float(b), floor=fl, highpass_hz=hp)
            for gl, gh, (a, b), fl, hp in itertools.product(*GRID.values())]
    df = pd.DataFrame(rows)
    df.insert(0, "cfg", [f"g{r.gamma_lo:g}-{r.gamma_hi:g}_ab{r.snr_lo:g}-{r.snr_hi:g}_f{r.floor:g}_hp{r.highpass_hz:g}"
                         for r in df.itertuples()])
    return df


def is_identity(r) -> bool:
    return r["gamma_lo"] == 1.0 and r["gamma_hi"] == 1.0 and r["floor"] == 0.0 and r["highpass_hz"] == 0.0


def model_of(cfg):
    p = cfg["params"]
    return gm._model(p["weights"], p.get("device", "cpu"))


def score(clean, y, sr=16000):
    pcm, _ = to_pcm16(y)
    y16 = pcm.astype(np.float64) / 32768.0  # exactly what main.py writes
    return pesq_wb(clean, y16, sr)[0], snr_db(clean, y16)


# ---------------------------------------------------------------- identity
def stage_identity(cfg, root, a) -> None:
    ref_src = subprocess.run(["git", "show", f"{a.ref_rev}:src/nrc/methods/gru_mask.py"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout
    tmp = REPO_ROOT / "logs" / "_gru_mask_ref.py"
    tmp.write_text(ref_src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("nrc.methods._gru_mask_ref", tmp)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    params = {k: v for k, v in cfg["params"].items() if k != "mask_post"}
    f_new = get_method("gru_mask", **params)
    f_id = get_method("gru_mask", **params, mask_post={"gamma_lo": 1.0, "gamma_hi": 1.0, "floor": 0.0,
                                                       "highpass_hz": 0.0})
    man = stratified_subset(load_manifest(root, "validation"), 20)
    rec = pd.read_csv(REPO_ROOT / "outputs" / a.ref_name / "eval_validation.csv").set_index("segment_id")
    sroot = split_root(root, "validation")
    d_ref = d_id = d_pesq = 0.0
    for _, row in man.iterrows():
        x = read_row(sroot, row, "noisy")
        y_ref = ref.enhance(x, 16000, **params)
        y_new, y_id = f_new(x), f_id(x)
        d_ref, d_id = max(d_ref, np.abs(y_new - y_ref).max()), max(d_id, np.abs(y_id - y_new).max())
        p, _ = score(read_row(sroot, row, "clean"), y_new)
        d_pesq = max(d_pesq, abs(p - rec.pesq[row["segment_id"]]))
    tmp.unlink()
    print(f"identity check on {len(man)} validation files ({a.ref_rev} code, {a.ref_name} rows):")
    print(f"  max|off - {a.ref_rev}| = {d_ref:.3g}   max|identity block - off| = {d_id:.3g}   "
          f"max|PESQ16 - recorded| = {d_pesq:.3g}")
    print("  PASS" if d_ref == 0 and d_id == 0 and d_pesq == 0 else "  FAIL")


# ---------------------------------------------------------------- cache
def stage_cache(cfg, root, man, a) -> None:
    import torch
    out = CACHE / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    if cfg["params"].get("level_norm"):
        raise SystemExit("cache stage assumes level_norm off (the M, X of the unscaled input)")
    model = model_of(cfg)
    sroot = split_root(root, a.split)
    with torch.inference_mode():
        for _, row in man.iterrows():
            x = read_row(sroot, row, "noisy")
            wave = torch.from_numpy(x.astype(np.float32))[None]
            spec, mask, length = gm.mask_and_spec(model, wave)
            np.savez(out / f"{row['segment_id']}.npz", spec=spec[0].numpy(), mask=mask[0].numpy(),
                     length=length, n=x.size)
    print(f"cached {len(man)} files -> {out}")


def _load(cfg_name, sid):
    import torch
    z = np.load(CACHE / cfg_name / f"{sid}.npz")
    return torch.from_numpy(z["spec"])[None], torch.from_numpy(z["mask"])[None], int(z["length"]), int(z["n"])


# ---------------------------------------------------------------- grid / frame workers
def _grid_file(task):
    import torch
    cfg, root, row, points = task
    torch.set_num_threads(1)
    model = model_of(cfg)
    spec, mask, length, n = _load(cfg["name"], row["segment_id"])
    clean = read_row(root, row, "clean")  # scoring only
    snr_f = gm.mask_snr_db(mask[0].numpy(), spec[0].numpy(), model.hop, 16000, "file")
    memo, out = {}, []
    with torch.inference_mode():
        for p in points:
            gamma = float(gm.gamma_of_snr(snr_f, p["gamma_lo"], p["gamma_hi"], p["snr_lo"], p["snr_hi"]))
            key = (round(gamma, 12), p["floor"], p["highpass_hz"])
            if key not in memo:
                y = gm.synth(model, spec, gm.post_mask(mask, gamma, p["floor"]), length, n)[0].double().numpy()
                memo[key] = score(clean, gm.highpass(y, p["highpass_hz"], 16000))
            out.append((row["segment_id"], p["cfg"], snr_f, gamma, *memo[key]))
    return out


def _frame_file(task):
    import torch
    cfg, root, row, p = task
    torch.set_num_threads(1)
    model = model_of(cfg)
    spec, mask, length, n = _load(cfg["name"], row["segment_id"])
    clean = read_row(root, row, "clean")
    s = gm.mask_snr_db(mask[0].numpy(), spec[0].numpy(), model.hop, 16000, "frame", p["smooth_s"])
    gamma = gm.gamma_of_snr(s, p["gamma_lo"], p["gamma_hi"], p["snr_lo"], p["snr_hi"])
    with torch.inference_mode():
        y = gm.synth(model, spec, gm.post_mask(mask, gamma, p["floor"]), length, n)[0].double().numpy()
    pe, sn = score(clean, gm.highpass(y, p["highpass_hz"], 16000))
    return row["segment_id"], p["cfg"], float(np.mean(s)), float(np.mean(gamma)), pe, sn


def run_pool(fn, tasks, jobs):
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(fn, tasks, chunksize=2))
    return [fn(t) for t in tasks]


COLS = ["segment_id", "cfg", "snr_est_db", "gamma", "pesq", "snr_db"]


def stage_grid(cfg, root, man, a) -> None:
    pts = grid_points().to_dict("records")
    sroot = split_root(root, a.split)
    res = run_pool(_grid_file, [(cfg, sroot, r.to_dict(), pts) for _, r in man.iterrows()], a.jobs)
    df = pd.DataFrame([x for f in res for x in f], columns=COLS)
    df.to_csv(REPO_ROOT / "logs" / f"r2_mpost_grid_{cfg['name']}_{a.split}{a.subset}.csv", index=False)
    print(f"grid: {len(pts)} points x {len(man)} files, PESQ NaN {int(df.pesq.isna().sum())}")


def best_point(cfg, a) -> dict:
    df = pd.read_csv(REPO_ROOT / "logs" / f"r2_mpost_grid_{cfg['name']}_{a.split}{a.subset}.csv")
    best = df.groupby("cfg").pesq.mean().idxmax()
    return grid_points().set_index("cfg").loc[best].to_dict() | {"cfg": best}


def stage_frame(cfg, root, man, a) -> None:
    p = json.loads(a.point) if a.point else best_point(cfg, a)
    p = {**gm.MASK_POST_DEFAULTS, **p, "smooth_s": a.smooth_s}
    p["cfg"] = "frame_" + p["cfg"]
    sroot = split_root(root, a.split)
    res = run_pool(_frame_file, [(cfg, sroot, r.to_dict(), p) for _, r in man.iterrows()], a.jobs)
    pd.DataFrame(res, columns=COLS).to_csv(
        REPO_ROOT / "logs" / f"r2_mpost_frame_{cfg['name']}_{a.split}{a.subset}.csv", index=False)
    print(f"frame variant of {p['cfg']} (smooth {a.smooth_s} s) on {len(man)} files done")


# ---------------------------------------------------------------- report
def stage_report(cfg, root, man, a) -> None:
    tag = f"{cfg['name']}_{a.split}{a.subset}"
    df = pd.read_csv(REPO_ROOT / "logs" / f"r2_mpost_grid_{tag}.csv")
    pts = grid_points().set_index("cfg")
    wide = df.pivot(index="segment_id", columns="cfg", values="pesq")
    snr = df.pivot(index="segment_id", columns="cfg", values="snr_db")
    ident = [c for c in pts.index if is_identity(pts.loc[c])][0]
    base = wide[ident]
    rows = []
    for c in wide.columns:
        st = paired_stats(base, wide[c])
        rows.append({"cfg": c, **pts.loc[c].to_dict(), "pesq": wide[c].mean(), "snr_db": snr[c].mean(),
                     "delta": st["d_mean"], "se": st["d_se"], "z": st["d_z"]})
    tab = pd.DataFrame(rows).sort_values("pesq", ascending=False)
    tab.to_csv(REPO_ROOT / "results" / f"r2_mpost_grid_{tag}.csv", index=False, float_format="%.5f")
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print(f"== grid on {a.split} subset {a.subset} ({len(wide)} files, {len(wide.columns)} points) ==")
    print(f"identity ({ident}): PESQ {base.mean():.4f}  SNR {snr[ident].mean():.2f} dB")
    print(f"train-subset vs validation gap: {base.mean():.4f} - {VAL_PESQ} = {base.mean() - VAL_PESQ:+.4f}")
    print("\ntop 5 by mean PESQ (paired vs identity):")
    print(tab.head(5).to_string(index=False, float_format=fmt))
    meta = man.set_index("segment_id")
    best = tab.iloc[0]["cfg"]
    print(f"\nbest {best} per SNR (delta vs identity):")
    for s, idx in meta.groupby("target_snr_db").groups.items():
        st = paired_stats(base[idx], wide[best][idx])
        print(f"  {s:>4g} dB n={len(idx):3d} identity {base[idx].mean():.4f} delta {st['d_mean']:+.4f} "
              f"SE {st['d_se']:.4f} z {st['d_z']:+.2f}")
    # marginal views
    for k in ("gamma_lo", "gamma_hi", "floor", "highpass_hz"):
        print(f"\nbest mean PESQ by {k}: " + ", ".join(
            f"{v:g}: {tab[tab[k] == v].pesq.max():.4f}" for v in sorted(tab[k].unique())))
    fr = REPO_ROOT / "logs" / f"r2_mpost_frame_{tag}.csv"
    if fr.exists():
        f = pd.read_csv(fr).set_index("segment_id")
        name = f.cfg.iloc[0]
        src = name.removeprefix("frame_")
        print(f"\n== frame-level variant {name} ==")
        print(f"  file-level {src}: {wide[src].mean():.4f} | frame-level: {f.pesq.mean():.4f} SNR {f.snr_db.mean():.2f}")
        for lab, ref in (("vs identity", base), ("vs file-level", wide[src])):
            st = paired_stats(ref, f.pesq.reindex(ref.index))
            print(f"  {lab}: delta {st['d_mean']:+.4f} SE {st['d_se']:.4f} z {st['d_z']:+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("identity", "cache", "grid", "frame", "report"))
    ap.add_argument("--config", default="configs/dl/gru_v5b_off.yaml")
    ap.add_argument("--split", default="train", choices=("train", "validation", "val_remix"))
    ap.add_argument("--subset", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--point", default=None, help="frame stage: JSON grid point (default: grid best)")
    ap.add_argument("--smooth-s", type=float, default=0.25)
    ap.add_argument("--ref-rev", default="r2-prep", help="identity stage: code revision to compare against")
    ap.add_argument("--ref-name", default="gru_v5b_off", help="identity stage: outputs/<name> with eval rows")
    a = ap.parse_args()
    if a.jobs > 4:
        raise SystemExit("--jobs 4 or less")
    cfg = load_config(REPO_ROOT / a.config)
    root = resolve_data_root(None)
    if a.stage == "identity":
        stage_identity(cfg, root, a)
        return 0
    man = stratified_subset(load_manifest(root, a.split), a.subset)
    {"cache": stage_cache, "grid": stage_grid, "frame": stage_frame, "report": stage_report}[a.stage](cfg, root, man, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
