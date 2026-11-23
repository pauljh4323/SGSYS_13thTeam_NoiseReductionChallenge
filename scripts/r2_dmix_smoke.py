"""exp/r2-dmix CPU smoke test for configs/dl/gru_v8p2z{1,2}.yaml (no metrics rows; nothing is appended).
  1. Options off = old code: with gru_v8p2y's remix settings, nrc.data.remix.RemixData (new) and the exp/r2-dpesq
     version (git show 6f66fb9, = exp/r2-dpesq 791e7fa) give identical samples on 200 draws on the real train split, and
     2 training steps of train_dpesq.dpesq_loss (gru_v8p2x init, Adam lr 5e-5, clip 5, lambda 0.32) on the two batch
     streams give bitwise-equal losses (torch.equal).
  2. Real train data: dynamic mixing (gru_v8p2z2 settings) reproduces the drawn SNR within 0.1 dB on 100 draws;
     train-time rms_norm (gru_v8p2z1 and z2) gives mix RMS = target_rms on 100 draws.
  3. Per config: train_dpesq.py --max-steps 2 --val-subset 3 --train-subset 3 (2 losses finite, data options line);
     best.pt loads with weights_only=True.
  4. The rms_norm validation path (rms_norm_rows + enhance_rmsn) on 3 validation files equals
     methods.gru_mask.enhance(level_mode rms_norm, target_rms) with the same best.pt (max |diff| < 1e-6).
Usage: python scripts/r2_dmix_smoke.py --out <scratch dir> [--data-root ...]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

from _common import REPO_ROOT, load_config

from nrc.io import load_manifest, read_row, resolve_data_root, stratified_subset

CONFIGS = ("configs/dl/gru_v8p2z1.yaml", "configs/dl/gru_v8p2z2.yaml")
REF_CONFIG = "configs/dl/gru_v8p2y.yaml"
OLD_REMIX = "6f66fb9:src/nrc/data/remix.py"
STEP_RE = r"step (\d+): loss (\S+) \(cmag (\S+) mrstft (\S+) sisdr (\S+) pesq (\S+)\) tpesq_mos (\S+)"


def _run(cmd, label):
    print(f"{label} " + " ".join(str(c) for c in cmd[1:]), flush=True)
    t = time.perf_counter()
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = res.stdout.strip().splitlines()
    print("   " + "\n   ".join(out[-9:]) + f"\n   ({time.perf_counter() - t:.1f}s, exit {res.returncode})", flush=True)
    if res.returncode != 0:
        print(res.stderr[-3000:])
    return res.returncode == 0, res.stdout


def old_remix_class():
    src = subprocess.run(["git", "show", OLD_REMIX], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("remix_old")
    exec(compile(src.replace("from ..io import read_row", "from nrc.io import read_row"), OLD_REMIX, "exec"),
         mod.__dict__)
    return mod.RemixData


def snr(c, m):
    c, m = c.astype(np.float64), m.astype(np.float64)
    return 10 * np.log10(np.sum(c ** 2) / np.sum((m - c) ** 2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)

    import torch
    from nrc.data.remix import RemixData
    from nrc.methods.gru_mask import enhance
    from nrc.models.gru_mask import load_weights
    from nrc.tpesq import pesq_loss
    from train import loss_weights
    from train_dpesq import dpesq_loss, enhance_rmsn, rms_norm_rows, rms_target

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    root = resolve_data_root(args.data_root)
    extra = ["--data-root", args.data_root] if args.data_root else []
    ok_all = True
    train_df = load_manifest(root, "train")
    noise_csv = REPO_ROOT / "logs" / "noise_check.csv"

    # 1. options off == exp/r2-dpesq
    ref = load_config(REF_CONFIG)
    t = time.perf_counter()
    new = RemixData.from_manifest(root, train_df, noise_csv, **ref["train"]["remix"])
    Old = old_remix_class()
    old = Old(new.clean, new.noisy, None, **ref["train"]["remix"])
    old.noise_idx = new.noise_idx  # same noise_check mask (from_manifest of the old class reads the same csv)
    ra, rb = np.random.default_rng(0), np.random.default_rng(0)
    same = all(np.array_equal(x, y) for _ in range(200) for x, y in zip(old.sample(ra), new.sample(rb)))
    same &= ra.bit_generator.state == rb.bit_generator.state
    print(f"1a. {len(new)} train files; RemixData new vs exp/r2-dpesq, gru_v8p2y remix settings, 200 draws identical "
          f"(samples + RNG state): {same} ({time.perf_counter() - t:.1f}s)", flush=True)
    ok_all &= same
    lw, lam = loss_weights(ref["train"]), float(ref["train"]["lambda_pesq"])
    loss_fn = pesq_loss(1.0, 16000)
    losses = {}
    for tag, data in (("old", old), ("new", new)):
        torch.manual_seed(0)
        model = load_weights(REPO_ROOT / ref["train"]["init_from"])
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=float(ref["train"]["lr"]))
        rng, losses[tag] = np.random.default_rng(0), []
        for _ in range(2):
            c, n = data.batch(rng, 16)
            loss, _, _, _, _ = dpesq_loss(model, loss_fn, torch.from_numpy(c), torch.from_numpy(n), lw, lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(ref["train"]["grad_clip"]))
            opt.step()
            losses[tag].append(loss.detach())
    bit = all(torch.equal(a, b) for a, b in zip(losses["old"], losses["new"]))
    print(f"1b. gru_v8p2y training loss, 2 steps, old vs new data code: bitwise equal {bit} "
          f"({[f'{x.item():.10f}' for x in losses['new']]})", flush=True)
    ok_all &= bit

    # 2. real-data dynamic mixing SNR + rms_norm level
    for cfg_path in CONFIGS:
        cfg = load_config(cfg_path)
        rt = rms_target(cfg)
        d = RemixData(new.clean, new.noisy, None, rms_norm_target=rt, **cfg["train"]["remix"])
        d.noise_idx = new.noise_idx
        rng, errs, rms_err, n2 = np.random.default_rng(1), [], [], 0
        for _ in range(100):
            info = {}
            c, m = d.sample(rng, info)
            rms_err.append(abs(np.sqrt(np.mean(m.astype(np.float64) ** 2)) / rt - 1))
            if info:
                errs.append(abs(snr(c, m) - info["snr_db"]))
                n2 += info["noise2"] is not None
        ok = max(rms_err) < 1e-5 and (not errs or (len(errs) == 100 and max(errs) < 0.1))
        print(f"2. {cfg['name']}: rms_norm max rel err {max(rms_err):.2e}"
              + (f"; dynamic mixing 100/100 draws, max |SNR err| {max(errs):.4f} dB, second noise {n2}/100"
                 if errs else "; no dynamic mixing draws (p_dm 0)") + f": {ok}", flush=True)
        ok_all &= ok

    # 3. 2 steps per config + best.pt
    small = ["--val-subset", "3", "--train-subset", "3", "--device", "cpu", "--max-steps", "2"] + extra
    script = str(REPO_ROOT / "scripts" / "train_dpesq.py")
    best = None
    for cfg_path in CONFIGS:
        name = Path(cfg_path).stem
        ck = out / name
        ok, stdout = _run([sys.executable, script, "--config", cfg_path, "--ckpt-dir", str(ck)] + small, f"3. {name}:")
        steps = re.findall(STEP_RE, stdout)
        fin = ok and len(steps) == 2 and all(np.isfinite(float(s[1])) for s in steps)
        print(f"   2 steps finite: {fin} {[(s[0], s[1], s[5]) for s in steps]}")
        ok_all &= fin
        if ok:
            meta = torch.load(ck / "best.pt", map_location="cpu", weights_only=True)
            print(f"   best.pt weights_only=True OK: keys {sorted(meta)}, epoch {meta['epoch']}, "
                  f"pesq {meta['pesq']:.4f}")
            best = best or ck / "best.pt"

    # 4. training validation path == inference rms_norm
    if best is None:
        ok_all = False
    else:
        cfg = load_config(CONFIGS[0])
        rt = rms_target(cfg)
        rows = list(stratified_subset(load_manifest(root, "validation"), 3).iterrows())
        noisy = np.stack([read_row(root, r, "noisy").astype(np.float64) for _, r in rows])
        model = load_weights(best)
        est = enhance_rmsn(model, rms_norm_rows(noisy, rt), "cpu", 16)
        inf = np.stack([enhance(x, 16000, weights=str(best), device="cpu", level_norm=True, level_mode="rms_norm",
                                target_rms=rt) for x in noisy])
        diff = float(np.max(np.abs(est - inf)))
        ok = diff < 1e-6
        print(f"4. validation path vs methods.gru_mask.enhance(rms_norm) on 3 val files: max |diff| {diff:.2e}: {ok}")
        ok_all &= ok
    print("SMOKE " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
