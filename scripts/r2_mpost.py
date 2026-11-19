"""exp/r2-mpost: SNR-adaptive mask-strength post-process for gru_mask (tuning tool, no training).

Stages (run one at a time; long ones detached, logs in logs/). Every long stage is resumable: rerun the
same command after a kill and it skips finished work.
  identity  20 validation files: mask_post off == r2-prep code (bit-exact), mask_post identity block == off,
            16-bit PESQ == the recorded eval.py rows of the reference config.
  cache     run the model once per file; save its STFT X and mask M (logs/r2_mpost_cache/<config>/, one .npz
            per file; existing files are skipped).
  grid      gamma_lo x gamma_hi x (a, b) x floor x hpf on the cached M, X (16-bit PESQ, as main.py writes).
            One block of per-file rows is appended (and flushed) to --out per finished grid point; points
            already complete in --out are skipped. --points restricts the run (e.g. identity + best for the
            full-train confirmation). Identical (file, gamma, floor, hpf) evaluations are computed once.
  frame     frame-level SNR_est variant of one grid point (default: the grid's best); per-file rows appended.
  report    top-5 table with paired deltas vs identity, frame vs file, train-subset vs validation gap.
Workers load one cached file at a time (np.load per task); the main process never holds the cache.
Peak RSS (Windows PeakWorkingSetSize; ru_maxrss elsewhere) of every worker is logged.
The post-process code is the method's own (nrc.methods.gru_mask: mask_snr_db, gamma_of_snr, post_mask,
synth, highpass), so a grid point reproduces enhance(..., mask_post=...) exactly.
Usage: python scripts/r2_mpost.py --stage grid --jobs 2 > logs/r2_mpost_grid.log 2>&1
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
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
IDENTITY = "g1-1_ab0-10_f0_hp0"
VAL_PESQ = 2.6962  # gru_v5b_off validation (results/experiments.csv, r2-prep)
COLS = ["segment_id", "cfg", "snr_est_db", "gamma", "pesq", "snr_db"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_rss_mb() -> float:
    """Peak resident set of this process in MB."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2 ** 20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def grid_points() -> pd.DataFrame:
    rows = [dict(gamma_lo=gl, gamma_hi=gh, snr_lo=float(a), snr_hi=float(b), floor=fl, highpass_hz=hp)
            for gl, gh, (a, b), fl, hp in itertools.product(*GRID.values())]
    df = pd.DataFrame(rows)
    df.insert(0, "cfg", [f"g{r.gamma_lo:g}-{r.gamma_hi:g}_ab{r.snr_lo:g}-{r.snr_hi:g}_f{r.floor:g}_hp{r.highpass_hz:g}"
                         for r in df.itertuples()])
    # identity first: every later point's paired delta needs it
    return pd.concat([df[df.cfg == IDENTITY], df[df.cfg != IDENTITY]]).reset_index(drop=True)


def model_of(cfg):
    p = cfg["params"]
    return gm._model(p["weights"], p.get("device", "cpu"))


def score(clean, y, sr=16000):
    pcm, _ = to_pcm16(y)
    y16 = pcm.astype(np.float64) / 32768.0  # exactly what main.py writes
    return pesq_wb(clean, y16, sr)[0], snr_db(clean, y16)


def append_rows(path: Path, rows: list) -> None:
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        pd.DataFrame(rows, columns=COLS).to_csv(f, header=new, index=False)
        f.flush()
        os.fsync(f.fileno())


class RSS:
    """Peak RSS per worker pid, as reported by the workers."""
    def __init__(self):
        self.peak = {}

    def add(self, pid, mb):
        self.peak[pid] = max(self.peak.get(pid, 0.0), mb)

    def __str__(self):
        return ", ".join(f"pid {p}: {m:.0f} MB" for p, m in sorted(self.peak.items())) or "-"


def make_pool(jobs):
    return ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None


def pmap(pool, fn, tasks, chunksize=4):
    return pool.map(fn, tasks, chunksize=chunksize) if pool else map(fn, tasks)


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
def _cache_file(task):
    import torch
    cfg, root, row, threads = task
    torch.set_num_threads(threads)
    out = CACHE / cfg["name"] / f"{row['segment_id']}.npz"
    x = read_row(root, row, "noisy")
    with torch.inference_mode():
        spec, mask, length = gm.mask_and_spec(model_of(cfg), torch.from_numpy(x.astype(np.float32))[None])
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, spec=spec[0].numpy(), mask=mask[0].numpy(), length=length, n=x.size)
    os.replace(tmp, out)  # a kill never leaves a truncated cache file
    return os.getpid(), peak_rss_mb()


def stage_cache(cfg, root, man, a) -> None:
    if cfg["params"].get("level_norm"):
        raise SystemExit("cache stage assumes level_norm off (the M, X of the unscaled input)")
    out = CACHE / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    todo = [r.to_dict() for _, r in man.iterrows() if not (out / f"{r['segment_id']}.npz").exists()]
    log(f"cache: {len(man) - len(todo)} of {len(man)} present, {len(todo)} to do (jobs {a.jobs})")
    sroot, rss, pool = split_root(root, a.split), RSS(), make_pool(a.jobs)
    threads = max(1, 4 // a.jobs)
    for i, (pid, mb) in enumerate(pmap(pool, _cache_file, [(cfg, sroot, r, threads) for r in todo]), 1):
        rss.add(pid, mb)
        if i % 100 == 0 or i == len(todo):
            log(f"cache {i}/{len(todo)} | peak RSS {rss}")
    if pool:
        pool.shutdown()
    log(f"cached -> {out}")


def _load(cfg_name, sid):
    import torch
    with np.load(CACHE / cfg_name / f"{sid}.npz") as z:
        return (torch.from_numpy(z["spec"])[None], torch.from_numpy(z["mask"])[None],
                int(z["length"]), int(z["n"]))


# ---------------------------------------------------------------- grid workers
def _snr_file(task):
    cfg, sid = task
    spec, mask, _, _ = _load(cfg["name"], sid)
    hop = model_of(cfg).hop
    return sid, gm.mask_snr_db(mask[0].numpy(), spec[0].numpy(), hop, 16000, "file"), os.getpid(), peak_rss_mb()


def _eval_one(task):
    """One (file, gamma, floor, hpf) evaluation; gamma is a scalar (file) or per-frame params (frame)."""
    import torch
    cfg, root, row, gamma, floor, hpf, frame = task
    torch.set_num_threads(1)
    model = model_of(cfg)
    spec, mask, length, n = _load(cfg["name"], row["segment_id"])
    if frame is not None:  # frame-level SNR_est -> per-frame gamma
        s = gm.mask_snr_db(mask[0].numpy(), spec[0].numpy(), model.hop, 16000, "frame", frame["smooth_s"])
        g = gm.gamma_of_snr(s, frame["gamma_lo"], frame["gamma_hi"], frame["snr_lo"], frame["snr_hi"])
        snr_e, gamma = float(np.mean(s)), g
    else:
        snr_e, g = np.nan, gamma
    with torch.inference_mode():
        y = gm.synth(model, spec, gm.post_mask(mask, g, floor), length, n)[0].double().numpy()
    pe, sn = score(read_row(root, row, "clean"), gm.highpass(y, hpf, 16000))
    return row["segment_id"], snr_e, float(np.mean(gamma)), pe, sn, os.getpid(), peak_rss_mb()


def file_snrs(cfg, man, pool, rss, path: Path) -> pd.Series:
    have = pd.read_csv(path).set_index("segment_id").snr_est_db if path.exists() else pd.Series(dtype=float)
    todo = [s for s in man.segment_id if s not in have.index]
    rows = []
    for sid, s, pid, mb in pmap(pool, _snr_file, [(cfg, s) for s in todo], 16):
        rss.add(pid, mb)
        rows.append((sid, s))
    if rows:
        have = pd.concat([have, pd.DataFrame(rows, columns=["segment_id", "snr_est_db"])
                          .set_index("segment_id").snr_est_db])
        have.rename("snr_est_db").to_csv(path)
    return have.loc[list(man.segment_id)]


def read_done(path: Path, n_files: int):
    """Existing rows of complete points (a point with fewer than n_files rows is redone)."""
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path)
    cnt = df.groupby("cfg").segment_id.nunique()
    done = cnt[cnt >= n_files].index
    if len(done) < len(cnt):  # drop a partial block and rewrite the file
        df = df[df.cfg.isin(done)]
        df.to_csv(path, index=False)
    return df


def stage_grid(cfg, root, man, a) -> None:
    pts = grid_points()
    if a.points:
        names = [best_point(cfg, a)["cfg"] if p == "best" else IDENTITY if p == "identity" else p
                 for p in a.points.split(",")]
        pts = pts[pts.cfg.isin(names)]
    out = REPO_ROOT / a.out
    done_df = read_done(out, len(man))
    done = set(done_df.cfg)
    pts_floor = grid_points().set_index("cfg")
    memo = {(r.segment_id, round(r.gamma, 12), pts_floor.floor[r.cfg], pts_floor.highpass_hz[r.cfg]): (r.pesq, r.snr_db)
            for r in done_df.itertuples()}
    del done_df
    todo = [p for p in pts.to_dict("records") if p["cfg"] not in done]
    log(f"grid -> {a.out}: {len(pts)} points x {len(man)} files ({a.split} subset {a.subset}); "
        f"{len(pts) - len(todo)} done, {len(todo)} to do; memo {len(memo)}; jobs {a.jobs}; main RSS {peak_rss_mb():.0f} MB")
    rss, pool = RSS(), make_pool(a.jobs)
    sroot = split_root(root, a.split)
    snr = file_snrs(cfg, man, pool, rss, REPO_ROOT / "logs" / f"r2_mpost_snr_{cfg['name']}_{a.split}{a.subset}.csv")
    rows_by_sid = {r["segment_id"]: r.to_dict() for _, r in man.iterrows()}
    t0 = time.perf_counter()
    for i, p in enumerate(todo, 1):
        gam = {s: round(float(gm.gamma_of_snr(snr[s], p["gamma_lo"], p["gamma_hi"], p["snr_lo"], p["snr_hi"])), 12)
               for s in man.segment_id}
        need = sorted({(s, gam[s]) for s in man.segment_id
                       if (s, gam[s], p["floor"], p["highpass_hz"]) not in memo})
        tasks = [(cfg, sroot, rows_by_sid[s], g, p["floor"], p["highpass_hz"], None) for s, g in need]
        for sid, _, g, pe, sn, pid, mb in pmap(pool, _eval_one, tasks):
            memo[(sid, round(g, 12), p["floor"], p["highpass_hz"])] = (pe, sn)
            rss.add(pid, mb)
        block = [(s, p["cfg"], snr[s], gam[s], *memo[(s, gam[s], p["floor"], p["highpass_hz"])])
                 for s in man.segment_id]
        append_rows(out, block)
        log(f"point {len(pts) - len(todo) + i}/{len(pts)} {p['cfg']}: PESQ {np.nanmean([b[4] for b in block]):.4f} "
            f"| new evals {len(tasks)} | {time.perf_counter() - t0:.0f} s | peak RSS {rss}")
    if pool:
        pool.shutdown()
    log(f"grid done; PESQ NaN in file: {int(pd.read_csv(out).pesq.isna().sum())}; peak RSS per worker: {rss}; "
        f"main {peak_rss_mb():.0f} MB")


def best_point(cfg, a) -> dict:
    df = pd.read_csv(REPO_ROOT / a.grid_csv)
    best = df.groupby("cfg").pesq.mean().idxmax()
    return grid_points().set_index("cfg").loc[best].to_dict() | {"cfg": best}


def stage_frame(cfg, root, man, a) -> None:
    p = json.loads(a.point) if a.point else best_point(cfg, a)
    p = {**gm.MASK_POST_DEFAULTS, **p, "smooth_s": a.smooth_s}
    name = f"frame{a.smooth_s:g}_" + p["cfg"]
    out = REPO_ROOT / a.out
    have = set(pd.read_csv(out).query("cfg == @name").segment_id) if out.exists() else set()
    sroot = split_root(root, a.split)
    tasks = [(cfg, sroot, r.to_dict(), None, p["floor"], p["highpass_hz"], p)
             for _, r in man.iterrows() if r["segment_id"] not in have]
    log(f"frame {name} -> {a.out}: {len(have)} done, {len(tasks)} to do (jobs {a.jobs})")
    rss, pool, buf = RSS(), make_pool(a.jobs), []
    for i, (sid, s, g, pe, sn, pid, mb) in enumerate(pmap(pool, _eval_one, tasks), 1):
        rss.add(pid, mb)
        buf.append((sid, name, s, g, pe, sn))
        if len(buf) >= 50 or i == len(tasks):
            append_rows(out, buf)
            buf = []
            log(f"frame {i}/{len(tasks)} | peak RSS {rss}")
    if pool:
        pool.shutdown()
    log(f"frame done; peak RSS per worker: {rss}")


# ---------------------------------------------------------------- report
def stage_report(cfg, root, man, a) -> None:
    df = pd.read_csv(REPO_ROOT / a.grid_csv)
    pts = grid_points().set_index("cfg")
    wide = df.pivot(index="segment_id", columns="cfg", values="pesq")
    snr = df.pivot(index="segment_id", columns="cfg", values="snr_db")
    base = wide[IDENTITY]
    rows = []
    for c in wide.columns:
        st = paired_stats(base, wide[c])
        rows.append({"cfg": c, **pts.loc[c].to_dict(), "pesq": wide[c].mean(), "snr_db": snr[c].mean(),
                     "delta": st["d_mean"], "se": st["d_se"], "z": st["d_z"]})
    tab = pd.DataFrame(rows).sort_values("pesq", ascending=False)
    tag = f"{cfg['name']}_{a.split}{a.subset}"
    tab.to_csv(REPO_ROOT / "results" / f"r2_mpost_grid_{tag}.csv", index=False, float_format="%.5f")
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print(f"== {a.grid_csv}: {a.split} subset {a.subset} ({len(wide)} files, {len(wide.columns)} points) ==")
    print(f"identity ({IDENTITY}): PESQ {base.mean():.4f}  SNR {snr[IDENTITY].mean():.2f} dB")
    print(f"train vs validation gap (identity): {base.mean():.4f} - {VAL_PESQ} = {base.mean() - VAL_PESQ:+.4f}")
    print("\ntop 5 by mean PESQ (paired vs identity):")
    print(tab.head(5).drop(columns="cfg").to_string(index=False, float_format=fmt))
    meta = man.set_index("segment_id")
    best = tab.iloc[0]["cfg"]
    print(f"\nbest {best} per SNR (delta vs identity):")
    for s, idx in meta.groupby("target_snr_db").groups.items():
        st = paired_stats(base[idx], wide[best][idx])
        print(f"  {s:>4g} dB n={len(idx):3d} identity {base[idx].mean():.4f} best {wide[best][idx].mean():.4f} "
              f"delta {st['d_mean']:+.4f} SE {st['d_se']:.4f} z {st['d_z']:+.2f}")
    for k in ("gamma_lo", "gamma_hi", "floor", "highpass_hz"):
        print(f"best mean PESQ by {k}: " + ", ".join(
            f"{v:g}: {tab[tab[k] == v].pesq.max():.4f}" for v in sorted(tab[k].unique())))
    fr = REPO_ROOT / a.out
    if a.out != a.grid_csv and fr.exists():
        f = pd.read_csv(fr)
        for name, g in f.groupby("cfg"):
            g = g.set_index("segment_id")
            src = name.split("_", 1)[1]
            print(f"\n== frame-level variant {name} (n={len(g)}) ==")
            print(f"  file-level {src}: {wide[src].mean():.4f} | frame-level: {g.pesq.mean():.4f} "
                  f"SNR {g.snr_db.mean():.2f} dB")
            for lab, ref in (("vs identity", base), ("vs file-level", wide[src])):
                st = paired_stats(ref, g.pesq.reindex(ref.index))
                print(f"  {lab}: delta {st['d_mean']:+.4f} SE {st['d_se']:.4f} z {st['d_z']:+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("identity", "cache", "grid", "frame", "report"))
    ap.add_argument("--config", default="configs/dl/gru_v5b_off.yaml")
    ap.add_argument("--split", default="train", choices=("train", "validation", "val_remix"))
    ap.add_argument("--subset", type=int, default=400, help="stratified subset size; 0 = full split")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=None, help="per-file rows CSV (grid default logs/r2_mpost_grid.csv, "
                                                "frame default logs/r2_mpost_frame.csv)")
    ap.add_argument("--grid-csv", default="logs/r2_mpost_grid.csv", help="grid rows used for 'best' / report")
    ap.add_argument("--points", default=None, help="grid: comma list of point names, 'identity', 'best'")
    ap.add_argument("--point", default=None, help="frame stage: JSON grid point (default: grid best)")
    ap.add_argument("--smooth-s", type=float, default=0.25)
    ap.add_argument("--ref-rev", default="r2-prep", help="identity stage: code revision to compare against")
    ap.add_argument("--ref-name", default="gru_v5b_off", help="identity stage: outputs/<name> with eval rows")
    a = ap.parse_args()
    if a.jobs > 4:
        raise SystemExit("--jobs 4 or less")
    a.out = a.out or {"grid": "logs/r2_mpost_grid.csv", "frame": "logs/r2_mpost_frame.csv",
                      "report": "logs/r2_mpost_frame.csv"}.get(a.stage, "")
    cfg = load_config(REPO_ROOT / a.config)
    root = resolve_data_root(None)
    if a.stage == "identity":
        stage_identity(cfg, root, a)
        return 0
    man = stratified_subset(load_manifest(root, a.split), a.subset or None)
    {"cache": stage_cache, "grid": stage_grid, "frame": stage_frame, "report": stage_report}[a.stage](cfg, root, man, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
