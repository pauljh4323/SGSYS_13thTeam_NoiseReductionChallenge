"""exp/r2-dmix CPU smoke test for configs/dl/gru_v8p2z3{a,b}.yaml (clean-side augmentation; no metrics rows).
  1. Options off = gru_v8p2z2: with gru_v8p2z2's remix settings + rms_norm, nrc.data.remix.RemixData (new) and the
     version before the clean augmentation (git show 363c4e5) give identical samples + RNG state on 200 real draws,
     and 2 training steps of train_dpesq.dpesq_loss (gru_v8p2z2 init, Adam lr 5e-5, clip 5) give bitwise-equal losses.
  2. Real train data, z3a/z3b settings, 100 draws: lengths unchanged, finite, SNR against the augmented clean within
     0.1 dB of the drawn SNR, mix RMS = target_rms; speed / EQ draw counts.
  3. Per config: train_dpesq.py --max-steps 2 --val-subset 3 --train-subset 3 (2 losses finite); best.pt loads with
     weights_only=True. Also reports the data-loader time per batch of 16 (z2 vs z3a vs z3b settings).
Usage: python scripts/r2_caug_smoke.py --out <scratch dir> [--data-root ...]
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
from r2_dmix_smoke import STEP_RE, _run, snr

from nrc.io import load_manifest, resolve_data_root

CONFIGS = ("configs/dl/gru_v8p2z3a.yaml", "configs/dl/gru_v8p2z3b.yaml")
REF_CONFIG = "configs/dl/gru_v8p2z2.yaml"
OLD_REMIX = "363c4e5:src/nrc/data/remix.py"


def old_remix_class():
    src = subprocess.run(["git", "show", OLD_REMIX], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("remix_old")
    exec(compile(src.replace("from ..io import read_row", "from nrc.io import read_row"), OLD_REMIX, "exec"),
         mod.__dict__)
    return mod.RemixData


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)

    import torch
    from nrc.data.remix import RemixData
    from nrc.models.gru_mask import load_weights
    from nrc.tpesq import pesq_loss
    from train import loss_weights
    from train_dpesq import dpesq_loss, rms_target

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    root = resolve_data_root(args.data_root)
    extra = ["--data-root", args.data_root] if args.data_root else []
    ok_all = True
    train_df = load_manifest(root, "train")
    noise_csv = REPO_ROOT / "logs" / "noise_check.csv"

    # 1. options off == gru_v8p2z2 (code before the clean augmentation)
    ref = load_config(REF_CONFIG)
    rt = rms_target(ref)
    t = time.perf_counter()
    new = RemixData.from_manifest(root, train_df, noise_csv, rms_norm_target=rt, **ref["train"]["remix"])
    old = old_remix_class()(new.clean, new.noisy, None, rms_norm_target=rt, **ref["train"]["remix"])
    old.noise_idx = new.noise_idx
    ra, rb = np.random.default_rng(0), np.random.default_rng(0)
    same = all(np.array_equal(x, y) for _ in range(200) for x, y in zip(old.sample(ra), new.sample(rb)))
    same &= ra.bit_generator.state == rb.bit_generator.state
    print(f"1a. {len(new)} train files x {new.length}; gru_v8p2z2 settings, new vs pre-augmentation RemixData, 200 draws "
          f"identical (samples + RNG state): {same} ({time.perf_counter() - t:.1f}s)", flush=True)
    ok_all &= same
    lw, lam = loss_weights(ref["train"]), float(ref["train"]["lambda_pesq"])
    loss_fn = pesq_loss(1.0, 16000)
    losses = {}
    for tag, data in (("old", old), ("new", new)):
        torch.manual_seed(0)
        model = load_weights(REPO_ROOT / "weights" / "gru_v8p2z2.pt")
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
    print(f"1b. gru_v8p2z2 training loss, 2 steps, options off vs pre-augmentation code: bitwise equal {bit} "
          f"({[f'{x.item():.10f}' for x in losses['new']]})", flush=True)
    ok_all &= bit

    # 2. real-data draws with the z3 settings + loader time
    def batch_time(d):
        rng = np.random.default_rng(5)
        d.batch(rng, 16)  # warm the filter cache
        t = time.perf_counter()
        for _ in range(10):
            d.batch(rng, 16)
        return (time.perf_counter() - t) / 10

    print(f"2. loader time per batch of 16, gru_v8p2z2 settings: {batch_time(new) * 1e3:.0f} ms")
    for cfg_path in CONFIGS:
        cfg = load_config(cfg_path)
        d = RemixData(new.clean, new.noisy, None, rms_norm_target=rms_target(cfg), **cfg["train"]["remix"])
        d.noise_idx = new.noise_idx
        rng, errs, rms_err, ns, ne, fin = np.random.default_rng(1), [], [], 0, 0, True
        for _ in range(100):
            info = {}
            c, m = d.sample(rng, info)
            fin &= c.shape == m.shape == (new.length,) and bool(np.all(np.isfinite(c)) and np.all(np.isfinite(m)))
            rms_err.append(abs(np.sqrt(np.mean(m.astype(np.float64) ** 2)) / rms_target(cfg) - 1))
            errs.append(abs(snr(c, m) - info["snr_db"]))
            ns += "speed" in info
            ne += "eq" in info
        ok = fin and max(errs) < 0.1 and max(rms_err) < 1e-5 and 25 < ns < 75 and (ne == 0 if d.p_eq == 0 else 25 < ne < 75)
        print(f"   {cfg['name']}: 100 draws, length/finite {fin}, max |SNR err| {max(errs):.4f} dB, rms rel err "
              f"{max(rms_err):.1e}, speed {ns}/100, EQ {ne}/100, loader {batch_time(d) * 1e3:.0f} ms/batch: {ok}",
              flush=True)
        ok_all &= ok

    # 3. 2 steps per config + best.pt
    small = ["--val-subset", "3", "--train-subset", "3", "--device", "cpu", "--max-steps", "2"] + extra
    script = str(REPO_ROOT / "scripts" / "train_dpesq.py")
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
            print(f"   best.pt weights_only=True OK: keys {sorted(meta)}, epoch {meta['epoch']}, pesq {meta['pesq']:.4f}")
        else:
            ok_all = False
    print("SMOKE " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
