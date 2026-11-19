"""exp/r2-dpesq CPU smoke test for configs/dl/gru_v8p{1,2,3}.yaml (no metrics rows; nothing is appended).
  1. lambda_pesq calibration at gru_v5b init: over 8 seed-0 remix batches (B 16), ||grad(cmag + 0.5 MR-STFT +
     0.3 (-SI-SDR))|| / ||grad(PESQ term)|| w.r.t. all model parameters; lambda = mean ratio. Checks that every
     config's lambda_pesq equals it rounded to 2 significant figures. --calibrate-only prints it and exits.
  2. Bit-identity: with lambda_pesq 0 and gru_v8p1's regression weights (= gru_v5b's), train_dpesq.dpesq_loss equals
     scripts/train.batch_loss (the gru_v5b training loss) on the same batch (torch.equal).
  3. Per variant: train_dpesq.py --max-steps 2 --val-subset 3 --train-subset 3 (2 steps, all logged losses finite,
     train_log row with train64_pesq / val_pesq / gap); best.pt loads with weights_only=True;
     main.py --config <tmp config with that best.pt> on 3 validation files keeps name, length, 16 kHz, mono.
  4. Resume (gru_v8p1): --max-steps 1, then --resume --max-steps 2; the resumed step 2 loss equals step 2 of the
     uninterrupted run in 3 (same batch, weights and Adam state; restore_torch_rng path).
Usage: python scripts/r2_dpesq_smoke.py --out <scratch dir> [--data-root ...] [--calibrate-only]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from _common import REPO_ROOT, load_config

from nrc.io import load_manifest, resolve_data_root, stratified_subset

CONFIGS = ("configs/dl/gru_v8p1.yaml", "configs/dl/gru_v8p2.yaml", "configs/dl/gru_v8p3.yaml")
STEP_RE = r"step (\d+): loss (\S+) \(cmag (\S+) mrstft (\S+) sisdr (\S+) pesq (\S+)\) tpesq_mos (\S+)"


def _run(cmd, label):
    print(f"{label} " + " ".join(str(c) for c in cmd[1:]), flush=True)
    t = time.perf_counter()
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = res.stdout.strip().splitlines()
    print("   " + "\n   ".join(out[-8:]) + f"\n   ({time.perf_counter() - t:.1f}s, exit {res.returncode})", flush=True)
    if res.returncode != 0:
        print(res.stderr[-3000:])
    return res.returncode == 0, res.stdout


def sig2(x: float) -> float:
    return float(f"{x:.2g}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--calibrate-only", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from nrc.data.remix import RemixData
    from nrc.models.gru_mask import count_params, load_weights
    from nrc.tpesq import pesq_loss
    from train import batch_loss, loss_weights
    from train_dpesq import calibrate, dpesq_loss, pesq_term

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    root = resolve_data_root(args.data_root)
    extra = ["--data-root", args.data_root] if args.data_root else []
    ok_all = True
    v5b = load_config("configs/dl/gru_v5b.yaml")
    lw5 = loss_weights(v5b["train"])
    data = RemixData.from_manifest(root, load_manifest(root, "train"), REPO_ROOT / "logs" / "noise_check.csv",
                                   **v5b["train"]["remix"])
    torch.manual_seed(0)
    g = load_weights(REPO_ROOT / "weights" / "gru_v5b.pt")
    g.train()
    loss_fn = pesq_loss(1.0, 16000)

    # 1. calibration
    t = time.perf_counter()
    cal = calibrate(g, loss_fn, data, np.random.default_rng(0), lw5, 16, 8, "cpu")
    lam = sig2(cal["lambda"])
    print(f"1. calibration at gru_v5b init, 8 batches x 16 (seed 0), {time.perf_counter() - t:.1f}s:\n"
          f"   ||grad regression (1.0 cmag + 0.5 MR-STFT + 0.3 (-SI-SDR))|| per batch {np.round(cal['reg'], 4).tolist()}\n"
          f"   ||grad PESQ term||                                      per batch {np.round(cal['pesq'], 4).tolist()}\n"
          f"   ratio mean {cal['lambda']:.4f} (min {cal['ratio'].min():.4f}, max {cal['ratio'].max():.4f}, "
          f"CV {cal['ratio'].std() / cal['ratio'].mean():.3f}) -> lambda_pesq = {lam}", flush=True)
    rng = np.random.default_rng(0)
    clean_b, noisy_b = data.batch(rng, 16)
    clean_t, noisy_t = torch.from_numpy(clean_b), torch.from_numpy(noisy_b)
    with torch.no_grad():
        est = g(noisy_t)
    ts = []
    for _ in range(3):
        e = est.clone().requires_grad_(True)
        t = time.perf_counter()
        pl, mos, _ = pesq_term(loss_fn, clean_t, e)
        pl.backward()
        ts.append(time.perf_counter() - t)
    print(f"   PESQ term forward+backward on 16 x 3 s (CPU, w.r.t. the wave): {np.median(ts):.3f}s; "
          f"batch tpesq MOS {mos.mean():.4f}")
    if args.calibrate_only:
        return 0
    for c in CONFIGS:
        got = float(load_config(c)["train"]["lambda_pesq"])
        print(f"   {Path(c).stem} lambda_pesq {got} == {lam}: {got == lam}")
        ok_all &= got == lam

    # 2. bit-identity with lambda 0
    cfg1 = load_config(CONFIGS[0])
    lw1 = loss_weights(cfg1["train"])
    with torch.no_grad():
        ref = batch_loss(g, clean_t, noisy_t, None, lw5)
        new, terms, mos, _, _ = dpesq_loss(g, loss_fn, clean_t, noisy_t, lw1, 0.0)
    bit = bool(torch.equal(ref, new))
    print(f"2. lambda 0, gru_v5b weights: train_dpesq loss == train.batch_loss (gru_v5b) bitwise: {bit} "
          f"({new.item():.10f} vs {ref.item():.10f}); weights v5b {lw5['cmag']}/{lw5['mrstft']}/{lw5['sisdr']} "
          f"= v8p1 {lw1['cmag']}/{lw1['mrstft']}/{lw1['sisdr']}; logged tpesq_mos {mos.mean():.4f}")
    ok_all &= bit

    # 3. variants
    names = [r["noisy_path"] for _, r in stratified_subset(load_manifest(root, "validation"), 3).iterrows()]
    inp = out / "input3"
    inp.mkdir()
    for rel in names:
        shutil.copy2(Path(root) / rel, inp / Path(rel).name)
    small = ["--val-subset", "3", "--train-subset", "3", "--device", "cpu"] + extra
    script = str(REPO_ROOT / "scripts" / "train_dpesq.py")
    step2 = {}
    for cfg_path in CONFIGS:
        cfg = load_config(cfg_path)
        name, lw = cfg["name"], loss_weights(cfg["train"])
        print(f"=== {name}: {lw['cmag']} cmag + {lw['mrstft']} MR-STFT + {lw['sisdr']} (-SI-SDR) + "
              f"{cfg['train']['lambda_pesq']} PESQ", flush=True)
        ck = out / name
        ok, stdout = _run([sys.executable, script, "--config", cfg_path, "--ckpt-dir", str(ck), "--max-steps", "2"]
                          + small, "3a.")
        steps = re.findall(STEP_RE, stdout)
        fin = len(steps) == 2 and all(np.isfinite(float(s[1])) and np.isfinite(float(s[5])) for s in steps)
        print(f"   2 steps finite (loss, PESQ term): {fin} {[(s[0], s[1], s[5], s[6]) for s in steps]}")
        ok_all &= ok and fin
        if not ok:
            continue
        step2[name] = steps[1][1] if len(steps) == 2 else None
        row = (ck / "train_log.csv").read_text(encoding="utf-8").strip().splitlines()
        print(f"   train_log.csv: {row[0]}\n                  {row[-1]}")
        m = load_weights(ck / "best.pt")
        meta = torch.load(ck / "best.pt", map_location="cpu", weights_only=True)
        print(f"3b. best.pt weights_only=True OK: keys {sorted(meta)}, epoch {meta['epoch']}, pesq {meta['pesq']:.4f}, "
              f"{count_params(m):,} params")
        tmp_cfg = out / f"{name}_smoke.yaml"
        tmp_cfg.write_text(yaml.safe_dump({"method": cfg["method"],
                                           "params": dict(cfg["params"], weights=str(ck / "best.pt"))}), encoding="utf-8")
        o = out / f"output_{name}"
        ok, _ = _run([sys.executable, str(REPO_ROOT / "main.py"), "--input", str(inp), "--output", str(o),
                      "--config", str(tmp_cfg)], "3c.")
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

    # 4. resume
    name = Path(CONFIGS[0]).stem
    ck = out / f"{name}_resume"
    ok1, _ = _run([sys.executable, script, "--config", CONFIGS[0], "--ckpt-dir", str(ck), "--max-steps", "1"] + small,
                  "4a.")
    ok2, stdout = _run([sys.executable, script, "--config", CONFIGS[0], "--ckpt-dir", str(ck), "--resume",
                        "--max-steps", "2"] + small, "4b.")
    steps = re.findall(STEP_RE, stdout)
    resumed = "resumed from" in stdout and len(steps) == 1 and steps[0][0] == "2"
    same = resumed and step2.get(name) is not None and steps[0][1] == step2[name]
    print(f"4. resume after 1 step: resumed {resumed}; step 2 loss {steps[0][1] if steps else None} == "
          f"uninterrupted {step2.get(name)}: {same}")
    ok_all &= ok1 and ok2 and same
    print("SMOKE " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
