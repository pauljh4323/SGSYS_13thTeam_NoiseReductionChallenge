"""exp/r2-mgan CPU smoke test for configs/dl/gru_v6m{1,2,3}.yaml (no metrics rows; nothing is appended).

  1. D forward/backward on one remixed batch (params, output range, finite gradients).
  2. PESQ pool timing: 4 validation files, enhanced-by-v5b and noisy vs clean, serial and in a pool of os.cpu_count().
  3. Bit-identity: with lambda_metric 0, gru_v5b weights and gru_v5b's loss weights, train_mgan.g_loss equals
     scripts/train.batch_loss (the gru_v5b training loss) on the same batch (torch.equal).
  4. D pretrain: train_mgan.py --pretrain-d --max-steps 2 (shared by the variants below).
  5. Per variant: train_mgan.py --max-steps 2 --val-subset 3 (2 G+D steps, all logged losses finite, train_log row);
     best.pt loads with weights_only=True; last.pt holds G, D, both optimizers and the replay buffer;
     main.py --config <tmp config with that best.pt> on 3 validation files keeps name, length, 16 kHz, mono.
Usage: python scripts/r2_mgan_smoke.py --out <scratch dir> [--data-root ...]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from _common import REPO_ROOT, load_config

from nrc.io import load_manifest, read_row, resolve_data_root, stratified_subset

CONFIGS = ("configs/dl/gru_v6m1.yaml", "configs/dl/gru_v6m2.yaml", "configs/dl/gru_v6m3.yaml")


def _run(cmd, label):
    print(f"{label} " + " ".join(str(c) for c in cmd[1:]), flush=True)
    t = time.perf_counter()
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = res.stdout.strip().splitlines()
    print("   " + "\n   ".join(out[-8:]) + f"\n   ({time.perf_counter() - t:.1f}s, exit {res.returncode})", flush=True)
    if res.returncode != 0:
        print(res.stderr[-3000:])
    return res.returncode == 0, res.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)
    import torch
    from nrc.data.remix import RemixData
    from nrc.models.gru_mask import count_params, load_weights
    from nrc.models.metric_disc import MetricDiscriminator
    from train import _pesq16, batch_loss, loss_weights
    from train_mgan import g_loss, mgan_cfg

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    root = resolve_data_root(args.data_root)
    ok_all = True
    extra = ["--data-root", args.data_root] if args.data_root else []

    cfg1 = load_config(CONFIGS[0])
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = RemixData.from_manifest(root, load_manifest(root, "train"), REPO_ROOT / "logs" / "noise_check.csv",
                                   **cfg1["train"]["remix"])
    clean_b, noisy_b = data.batch(rng, 16)
    clean_t, noisy_t = torch.from_numpy(clean_b), torch.from_numpy(noisy_b)
    g = load_weights(REPO_ROOT / "weights" / "gru_v5b.pt")

    # 1. D forward/backward
    disc = MetricDiscriminator()
    with torch.no_grad():
        est = g(noisy_t)
    pred = disc(est, clean_t)
    loss = torch.mean((pred - 0.5) ** 2)
    loss.backward()
    grads = [p.grad for p in disc.parameters() if p.requires_grad]
    good = all(gr is not None and torch.isfinite(gr).all() for gr in grads) and bool(((pred >= 0) & (pred <= 1)).all())
    print(f"1. D: {count_params(disc):,} params, input (16, 2, {disc.spec(clean_t).shape[1]}, {disc.spec(clean_t).shape[2]}), "
          f"out range [{pred.min().item():.4f}, {pred.max().item():.4f}], loss {loss.item():.5f}, "
          f"grads finite {good}")
    ok_all &= good

    # 2. PESQ pool timing on 4 validation files
    vdf = stratified_subset(load_manifest(root, "validation"), 4)
    vc = [read_row(root, r, "clean").astype(np.float64) for _, r in vdf.iterrows()]
    vn = [read_row(root, r, "noisy").astype(np.float32) for _, r in vdf.iterrows()]
    with torch.no_grad():
        ve = [g(torch.from_numpy(x)[None])[0].double().numpy() for x in vn]
    jobs = [(c, e) for c, e in zip(vc, ve)] + [(c, n.astype(np.float64)) for c, n in zip(vc, vn)]
    t = time.perf_counter()
    serial = [_pesq16(j) for j in jobs]
    t_serial = (time.perf_counter() - t) / len(jobs)
    n_cpu = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=n_cpu) as pool:
        list(pool.map(_pesq16, jobs))  # warm-up (process start)
        t = time.perf_counter()
        pooled = list(pool.map(_pesq16, jobs))
        t_pool = time.perf_counter() - t
        jobs32 = (jobs * 4)[:32]
        t = time.perf_counter()
        list(pool.map(_pesq16, jobs32))
        t_32 = time.perf_counter() - t
    same = np.allclose(serial, pooled, equal_nan=True) and np.isfinite(serial).all()
    print(f"2. PESQ ({len(jobs[0][0]) / 16000:.1f} s files): serial {t_serial * 1000:.0f} ms/call; pool of {n_cpu}: "
          f"{len(jobs)} calls {t_pool:.2f}s, 32 calls (one training step: 16 enh + 16 noisy) {t_32:.2f}s; "
          f"enh {np.mean(serial[:4]):.3f} / noisy {np.mean(serial[4:]):.3f}; pool == serial {same}")
    ok_all &= bool(same)

    # 3. bit-identity with lambda 0
    v5b = load_config("configs/dl/gru_v5b.yaml")
    g.train()
    lw5 = loss_weights(v5b["train"])
    lw1 = loss_weights(cfg1["train"])
    with torch.no_grad():
        ref = batch_loss(g, clean_t, noisy_t, None, lw5)
        new, terms, _, _ = g_loss(g, disc.eval(), clean_t, noisy_t, lw1, 0.0)
    bit = bool(torch.equal(ref, new))
    print(f"3. lambda 0, gru_v5b weights: train_mgan G loss == train.batch_loss (gru_v5b) bitwise: {bit} "
          f"({new.item():.10f} vs {ref.item():.10f}); weights v5b {lw5['cmag']}/{lw5['mrstft']}/{lw5['sisdr']} "
          f"= v6m1 {lw1['cmag']}/{lw1['mrstft']}/{lw1['sisdr']}")
    ok_all &= bit

    # 4. D pretrain (2 steps)
    d_pre = out / "d_pretrain.pt"
    ok, _ = _run([sys.executable, str(REPO_ROOT / "scripts" / "train_mgan.py"), "--config", CONFIGS[0],
                  "--pretrain-d", str(d_pre), "--max-steps", "2", "--device", "cpu"] + extra, "4.")
    ok_all &= ok and d_pre.is_file()

    # 5. variants
    names = [r["noisy_path"] for _, r in stratified_subset(load_manifest(root, "validation"), 3).iterrows()]
    inp = out / "input3"
    inp.mkdir()
    for rel in names:
        shutil.copy2(Path(root) / rel, inp / Path(rel).name)
    for cfg_path in CONFIGS:
        cfg = load_config(cfg_path)
        name, mg = cfg["name"], mgan_cfg(cfg["train"])
        lw = loss_weights(cfg["train"])
        print(f"=== {name}: {lw['cmag']} cmag + {lw['mrstft']} MR-STFT + {lw['sisdr']} (-SI-SDR) + "
              f"{mg['lambda']} (D-1)^2", flush=True)
        ck = out / name
        ok, stdout = _run([sys.executable, str(REPO_ROOT / "scripts" / "train_mgan.py"), "--config", cfg_path,
                           "--ckpt-dir", str(ck), "--d-init", str(d_pre), "--allow-unfinished-d", "--max-steps", "2",
                           "--val-subset", "3", "--device", "cpu"] + extra, "5a.")
        steps = re.findall(r"step (\d+): g_loss (\S+) .* d_loss (\S+)", stdout)
        fin = len(steps) == 2 and all(np.isfinite(float(a)) and np.isfinite(float(b)) for _, a, b in steps)
        print(f"   2 G+D steps finite: {fin} {steps}")
        ok_all &= ok and fin
        if not ok:
            continue
        print(f"   train_log.csv: {(ck / 'train_log.csv').read_text(encoding='utf-8').strip().splitlines()[-1]}")
        m = load_weights(ck / "best.pt")
        meta = torch.load(ck / "best.pt", map_location="cpu", weights_only=True)
        print(f"5b. best.pt weights_only=True OK: keys {sorted(meta)}, epoch {meta['epoch']}, pesq {meta['pesq']:.4f}, "
              f"{count_params(m):,} params")
        last = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
        has = all(k in last for k in ("model", "disc", "opt_g", "opt_d", "replay"))
        print(f"    last.pt full state: {has} (replay {last['replay']['x'].shape[0]} rows, "
              f"{(ck / 'last.pt').stat().st_size / 1e6:.1f} MB)")
        ok_all &= has
        tmp_cfg = out / f"{name}_smoke.yaml"
        tmp_cfg.write_text(yaml.safe_dump({"method": cfg["method"],
                                           "params": dict(cfg["params"], weights=str(ck / "best.pt"))}), encoding="utf-8")
        o = out / f"output_{name}"
        ok, _ = _run([sys.executable, str(REPO_ROOT / "main.py"), "--input", str(inp), "--output", str(o),
                      "--config", str(tmp_cfg)], "5c.")
        ok_all &= ok
        outs = sorted(p.name for p in o.iterdir()) if o.is_dir() else []
        good = outs == sorted(p.name for p in inp.iterdir())
        for p in sorted(inp.iterdir()):
            a, b = sf.info(str(p)), (sf.info(str(o / p.name)) if (o / p.name).is_file() else None)
            g_ok = b is not None and (a.frames, a.samplerate, a.channels) == (b.frames, b.samplerate, b.channels) \
                and b.samplerate == 16000 and b.channels == 1
            print(f"    {p.name}: in {a.frames} @ {a.samplerate} Hz {a.channels} ch -> "
                  + (f"out {b.frames} @ {b.samplerate} Hz {b.channels} ch: {'OK' if g_ok else 'FAIL'}" if b else "missing"))
            good &= g_ok
        ok_all &= good
    print("SMOKE " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
