"""exp/r2-mgan resume test for scripts/train_mgan.py (CPU; no metrics rows).

1. D pretrain 1 step; train configs/dl/gru_v6m2.yaml for 1 step (--max-steps 1) -> last.pt (epoch 1, step 1).
2. Three copies of that ckpt dir:
   plain      last.pt unchanged (CPU uint8 RNG state);
   int64      torch_rng stored as int64 (the type error Colab hit: a non-ByteTensor state);
   gpu_ckpt   torch_rng int64 + cuda_rng [int64 state] as saved on a CUDA runtime, resumed where CUDA is unavailable
              (must warn and skip the CUDA part).
   Before the fix, torch.set_rng_state on the int64 state raises "RNG state must be a torch.ByteTensor" (checked).
   restore_torch_rng must leave torch.get_rng_state() equal to the saved state in every case.
3. Each copy resumes with --resume --max-steps 2: it must continue at epoch 2 / step 2 with a finite loss, and all three
   step-2 losses must be identical (the RNG really was restored).
Usage: python scripts/r2_mgan_resume_test.py --out <scratch dir> [--data-root ...]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from _common import REPO_ROOT

CFG = "configs/dl/gru_v6m2.yaml"


def _run(args, extra):
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train_mgan.py"), "--config", CFG, "--device", "cpu",
           "--val-subset", "3"] + args + extra
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:], res.stderr[-3000:])
    return res.returncode, res.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args(argv)
    import torch
    from train_mgan import restore_torch_rng

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    extra = ["--data-root", a.data_root] if a.data_root else []
    ok = True
    d_pre = out / "d_pretrain.pt"
    rc, _ = _run(["--pretrain-d", str(d_pre), "--max-steps", "1"], extra)
    base = out / "base"
    rc2, so = _run(["--ckpt-dir", str(base), "--d-init", str(d_pre), "--allow-unfinished-d", "--max-steps", "1"], extra)
    s1 = re.findall(r"step 1: g_loss (\S+)", so)
    print(f"1. pretrain exit {rc}; 1-step run exit {rc2}, step 1 g_loss {s1}")
    ok &= rc == 0 and rc2 == 0 and len(s1) == 1

    ck = torch.load(base / "last.pt", map_location="cpu", weights_only=False)
    saved = ck["torch_rng"].clone()
    print(f"   last.pt: epoch {ck['epoch']}, step {ck['step']}, torch_rng {saved.dtype} {tuple(saved.shape)}, "
          f"cuda_rng {ck['cuda_rng']}, CUDA available {torch.cuda.is_available()}")
    try:
        torch.set_rng_state(saved.to(torch.int64))
        print("   old call torch.set_rng_state(int64 state): no error (unexpected)")
        ok = False
    except TypeError as e:
        print(f"   old call torch.set_rng_state(int64 state) -> TypeError: {e}")

    variants = {"plain": lambda c: c,
                "int64": lambda c: dict(c, torch_rng=c["torch_rng"].to(torch.int64)),
                "gpu_ckpt": lambda c: dict(c, torch_rng=c["torch_rng"].to(torch.int64),
                                           cuda_rng=[c["torch_rng"][:16].to(torch.int64)])}
    losses = {}
    for name, f in variants.items():
        d = out / name
        shutil.copytree(base, d)
        v = f(dict(ck))
        torch.save(v, d / "last.pt")
        torch.manual_seed(123)  # scramble, then restore
        restore_torch_rng(torch.load(d / "last.pt", map_location="cpu", weights_only=False))
        same = torch.equal(torch.get_rng_state(), saved)
        rc, so = _run(["--ckpt-dir", str(d), "--resume", "--max-steps", "2"], extra)
        resumed = re.findall(r"resumed from .*: epoch (\d+), step (\d+)", so)
        s2 = re.findall(r"step 2: g_loss (\S+)", so)
        ep = re.findall(r"epoch (\d+)/\d+ step (\d+)", so)
        warn = "CUDA is not available; skipped" in so
        good = rc == 0 and same and resumed == [("1", "1")] and len(s2) == 1 and np.isfinite(float(s2[0])) \
            and ep[-1] == ("2", "2") and (warn if name == "gpu_ckpt" else True)
        losses[name] = s2[0] if s2 else None
        print(f"2/3. {name}: restored state == saved {same}; exit {rc}; resumed at {resumed}; step 2 g_loss {s2}; "
              f"logged epoch/step {ep[-1] if ep else None}" + (f"; CUDA-skip warning {warn}" if name == "gpu_ckpt" else "")
              + f" -> {'OK' if good else 'FAIL'}")
        ok &= good
    equal = len(set(losses.values())) == 1 and None not in losses.values()
    print(f"   step-2 losses identical across variants: {equal} {losses}")
    ok &= equal
    print("RESUME TEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
