"""exp/r2-ext CPU smoke test for configs/dl/gru_v8p2z6{a,b}.yaml (ext_mix, selection sets, EMA; no metrics rows).
  1. Options off = gru_v8p2z3e: with z3e's settings, nrc.data.remix.RemixData (new) and the version before ext_mix
     (git show 4e23a58) give identical samples + RNG state on 200 real draws, and 2 training steps of
     train_dpesq.dpesq_loss (z3e init, Adam lr 5e-5, clip 5) give bitwise-equal losses. End to end: the old and the
     new scripts/train_dpesq.py --max-steps 2 print identical step lines for gru_v8p2z3e.
  2. Real package, z6a settings, 100 draws: length 48000, finite, SNR against the augmented clean within 0.1 dB of
     the drawn SNR, mix RMS = target_rms; clean / noise source shares, second-noise count; loader time per batch.
  3. Leakage: the package's noise index has no gen_dev / gen_test / excluded file, and check_noise_rels raises when
     one gen_dev file of each corpus is injected.
  4. Per config: train_dpesq.py --max-steps 2 --val-subset 3 --train-subset 3 (2 losses finite, selection sets
     logged); best.pt loads with weights_only=True and load_weights; for z6b the saved weights are the EMA weights
     (equal to last.pt's ema, different from last.pt's raw model).
Usage: python scripts/r2_ext_smoke.py --out <scratch dir> [--data-root ...] [--package ...]
"""
from __future__ import annotations

import argparse
import os
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

CONFIGS = ("configs/dl/gru_v8p2z6a.yaml", "configs/dl/gru_v8p2z6b.yaml")
REF_CONFIG = "configs/dl/gru_v8p2z3e.yaml"
BASE = "4e23a58"


def git_show(path: str) -> str:
    return subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout


def old_remix_class():
    src = git_show("src/nrc/data/remix.py")
    mod = types.ModuleType("remix_old")
    exec(compile(src.replace("from ..io import read_row", "from nrc.io import read_row"), "remix_old", "exec"),
         mod.__dict__)
    return mod.RemixData


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--package", default=None, help="default: train.ext_mix.package of gru_v8p2z6a")
    args = ap.parse_args(argv)

    import torch
    from nrc.data.extpool import CORPORA, SPLITS_DIR, check_noise_rels, split_sets
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

    # 1. options off == gru_v8p2z3e (code before ext_mix)
    ref = load_config(REF_CONFIG)
    rt = rms_target(ref)
    t = time.perf_counter()
    new = RemixData.from_manifest(root, train_df, noise_csv, rms_norm_target=rt, **ref["train"]["remix"])
    old = old_remix_class()(new.clean, new.noisy, None, rms_norm_target=rt, **ref["train"]["remix"])
    old.noise_idx = new.noise_idx
    ra, rb = np.random.default_rng(0), np.random.default_rng(0)
    same = all(np.array_equal(x, y) for _ in range(200) for x, y in zip(old.sample(ra), new.sample(rb)))
    same &= ra.bit_generator.state == rb.bit_generator.state
    print(f"1a. {len(new)} train files x {new.length}; gru_v8p2z3e settings, new vs {BASE} RemixData, 200 draws "
          f"identical (samples + RNG state): {same} ({time.perf_counter() - t:.1f}s)", flush=True)
    ok_all &= same
    lw, lam = loss_weights(ref["train"]), float(ref["train"]["lambda_pesq"])
    loss_fn = pesq_loss(1.0, 16000)
    losses = {}
    for tag, data in (("old", old), ("new", new)):
        torch.manual_seed(0)
        model = load_weights(REPO_ROOT / "weights" / "gru_v8p2z3e.pt")
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
    print(f"1b. gru_v8p2z3e training loss, 2 steps, options off vs {BASE} code: bitwise equal {bit} "
          f"({[f'{x.item():.10f}' for x in losses['new']]})", flush=True)
    ok_all &= bit
    small = ["--val-subset", "3", "--train-subset", "3", "--device", "cpu", "--max-steps", "2", "--threads", "4"] + extra
    old_script = out / "train_dpesq_old.py"
    old_script.write_text(git_show("scripts/train_dpesq.py"), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "scripts") + os.pathsep + str(REPO_ROOT / "src")}
    lines = {}
    for tag, script in (("old", old_script), ("new", REPO_ROOT / "scripts" / "train_dpesq.py")):
        res = subprocess.run([sys.executable, str(script), "--config", REF_CONFIG, "--ckpt-dir", str(out / f"z3e_{tag}")]
                             + small, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        lines[tag] = [re.sub(r"\s*\[[\d.]+s\]$", "", ln) for ln in res.stdout.splitlines() if re.search(STEP_RE, ln)]
        if res.returncode != 0:
            print(res.stderr[-2000:])
    e2e = len(lines["new"]) == 2 and lines["old"] == lines["new"]
    print(f"1c. train_dpesq.py gru_v8p2z3e --max-steps 2, {BASE} vs new: step lines identical {e2e}")
    for ln in lines["new"]:
        print("   " + ln.strip())
    ok_all &= e2e

    # 2. real package, z6a settings
    cfg_a = load_config(CONFIGS[0])
    ext_cfg = dict(cfg_a["train"]["ext_mix"])
    if args.package:
        ext_cfg["package"] = args.package
    t = time.perf_counter()
    d = RemixData(new.clean, new.noisy, None, rms_norm_target=rms_target(cfg_a), ext_mix=ext_cfg,
                  **cfg_a["train"]["remix"])
    d.noise_idx = new.noise_idx
    print(f"2. {d.ext.summary()} (load {time.perf_counter() - t:.1f}s)")
    rng, errs, rms_err, fin = np.random.default_rng(1), [], [], True
    n_libri = n_ext = n_second = n_speed = n_eq = 0
    corp = {c: 0 for c in CORPORA}
    for _ in range(100):
        info = {}
        c, m = d.sample(rng, info)
        fin &= c.shape == m.shape == (new.length,) and bool(np.all(np.isfinite(c)) and np.all(np.isfinite(m)))
        rms_err.append(abs(np.sqrt(np.mean(m.astype(np.float64) ** 2)) / rms_target(cfg_a) - 1))
        errs.append(abs(snr(c, m) - info["snr_db"]))
        n_libri += info["clean_src"][0] == "libri"
        for key in (info["noise"], info["noise2"]):
            if key is not None and key[0] == "ext":
                n_ext += key is info["noise"]
                corp[d.ext.noise.rows[key[1]]["corpus"]] += 1
        n_second += info["noise2"] is not None
        n_speed += "speed" in info
        n_eq += "eq" in info
    rng = np.random.default_rng(5)
    d.batch(rng, 16)
    t = time.perf_counter()
    for _ in range(10):
        d.batch(rng, 16)
    bt = (time.perf_counter() - t) / 10
    ok = (fin and max(errs) < 0.1 and max(rms_err) < 1e-5 and 30 <= n_libri <= 70 and 45 <= n_ext <= 75
          and 15 <= n_second <= 45 and min(corp.values()) > 0)
    print(f"   100 draws: length/finite {fin}, max |SNR err| {max(errs):.4f} dB, rms rel err {max(rms_err):.1e}, "
          f"LibriSpeech clean {n_libri}/100, external first noise {n_ext}/100, second noise {n_second}/100, external "
          f"noise per corpus {corp}, speed {n_speed}/100, EQ {n_eq}/100; loader {bt * 1e3:.0f} ms/batch of 16: {ok}",
          flush=True)
    ok_all &= ok

    # 3. leakage
    _, forbidden = split_sets()
    rels = [r["rel"] for r in d.ext.noise.rows]
    clean_set = not (set(rels) & forbidden)
    raised = 0
    for c in CORPORA:
        dev = (SPLITS_DIR / f"{c}_gen_dev.txt").read_text(encoding="utf-8").split()[0]
        try:
            check_noise_rels(rels + [dev])
        except AssertionError:
            raised += 1
    ok = clean_set and raised == len(CORPORA)
    print(f"3. leakage: {len(rels)} pool noise files, none in gen_dev/gen_test/excluded ({len(forbidden)} listed): "
          f"{clean_set}; injected gen_dev file raises: {raised}/{len(CORPORA)}: {ok}")
    ok_all &= ok

    # 4. 2 steps per config + best.pt
    pk = ["--ext-package", args.package] if args.package else []
    for cfg_path in CONFIGS:
        name = Path(cfg_path).stem
        ck = out / name
        ok, stdout = _run([sys.executable, str(REPO_ROOT / "scripts" / "train_dpesq.py"), "--config", cfg_path,
                           "--ckpt-dir", str(ck)] + small + pk, f"4. {name}:")
        steps = re.findall(STEP_RE, stdout)
        fin = ok and len(steps) == 2 and all(np.isfinite(float(s[1])) for s in steps) and "sel " in stdout
        print(f"   2 steps finite + selection logged: {fin} {[(s[0], s[1], s[5]) for s in steps]}")
        ok_all &= fin
        if not ok:
            ok_all = False
            continue
        meta = torch.load(ck / "best.pt", map_location="cpu", weights_only=True)
        load_weights(ck / "best.pt")
        print(f"   best.pt weights_only=True + load_weights OK: epoch {meta['epoch']}, sel pesq {meta['pesq']:.4f}, "
              f"val {meta.get('val_pesq')}, gu_dev {meta.get('gu_dev_pesq')}, gm_dev {meta.get('gm_dev_pesq')}, "
              f"ema_decay {meta.get('ema_decay')}")
        last = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
        best_sd = meta["model_state"]
        eq_model = all(torch.equal(best_sd[k], last["model"][k]) for k in best_sd)
        if load_config(cfg_path)["train"].get("ema_decay"):
            eq_ema = all(torch.equal(best_sd[k], last["ema"][k]) for k in best_sd)
            ok = eq_ema and not eq_model
            print(f"   EMA: best.pt == last.pt ema {eq_ema}, != raw model {not eq_model}: {ok}")
        else:
            ok = eq_model and last.get("ema") is None
            print(f"   no EMA: best.pt == last.pt model {eq_model}: {ok}")
        ok_all &= ok
    print("SMOKE " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
