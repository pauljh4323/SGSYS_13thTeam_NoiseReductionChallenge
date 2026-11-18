"""exp/r2-v7 CPU smoke test for configs/dl/gru_v7a.yaml and gru_v5b_s2.yaml (no metrics; nothing is appended).

Per config:
  A. fixed batch: model initialised exactly as scripts/train.py would (scratch: norm stats from the train noisy set;
     warm start: init_from), Adam at the config lr, 2 optimizer steps on ONE remixed batch with train.batch_loss;
     the loss is evaluated before step 1, after step 1 and after step 2 (pass: finite and lower after step 2
     than before step 1; strict monotonicity is printed too - from random init at lr 1e-3 the first Adam step
     overshoots on the SI-SDR term, see docs/exp_r2_v7.md).
  B. scripts/train.py --max-steps 2 --val-subset 3 into a scratch ckpt dir (the real training loop, writes best.pt/last.pt).
  C. best.pt via nrc.models.gru_mask.load_weights (torch.load weights_only=True); last.pt also tried weights_only=True.
  D. inference with the config's method/params (weights -> that best.pt) on 3 validation files:
     output written as 16-bit wav and read back; sample rate, channels and sample count must match the input.
Also checks that batch_loss reproduces the pre-r2-v7 inline loss of gru_v5b bit for bit on the same batch.
Usage: python scripts/r2_v7_smoke.py --out <scratch dir> [--data-root ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from _common import REPO_ROOT, load_config

from nrc.io import load_manifest, read_row, resolve_data_root, stratified_subset, write_wav_pcm16

CONFIGS = ("configs/dl/gru_v7a.yaml", "configs/dl/gru_v5b_s2.yaml")


def _init_model(cfg, data):
    import torch
    from nrc.models.gru_mask import GRUMask, load_partial
    from train import _norm_stats
    model = GRUMask(**cfg.get("model", {}))
    init_from = cfg["train"].get("init_from")
    if init_from:
        ck = torch.load(REPO_ROOT / init_from, map_location="cpu", weights_only=True)
        if ck["model_cfg"] == model.config:
            model.load_state_dict(ck["model_state"])
        else:
            load_partial(model, ck)
        how = f"init_from {init_from}"
    else:
        model.set_norm_stats(*_norm_stats(model, data.noisy, "cpu"))
        how = "random init (seeded) + train-set norm stats"
    return model, how


def _old_inline_loss(model, clean_t, noisy_t, tcfg):
    """The exp/r2-v6 inline loss for a magnitude-mask cmag+mrstft+sisdr config (lambda_ccplx 0)."""
    import torch
    from nrc.losses import cmag_loss, mrstft_loss, sisdr_loss
    c = float(tcfg.get("compress", 0.3))
    with torch.no_grad():
        target = model.stft(clean_t).abs()
    spec = model.stft(noisy_t)
    m = model.mask(spec.abs(), None)
    est_spec, est_mag = m * spec, m * spec.abs()
    loss = cmag_loss(est_mag, target, c)
    wave = model.istft(est_spec, length=noisy_t.shape[-1])
    loss = loss + float(tcfg["lambda_mrstft"]) * mrstft_loss(wave, clean_t)
    return loss + float(tcfg["lambda_sisdr"]) * sisdr_loss(wave, clean_t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)
    import torch
    from nrc.data.remix import RemixData
    from nrc.methods import get_method
    from nrc.models.gru_mask import count_params, load_weights
    from train import batch_loss, loss_weights

    out = Path(args.out)
    root = resolve_data_root(args.data_root)
    train_df = load_manifest(root, "train")
    val_df = stratified_subset(load_manifest(root, "validation"), 3)
    ok_all = True
    for cfg_path in CONFIGS:
        cfg = load_config(cfg_path)
        tcfg = cfg["train"]
        name = cfg["name"]
        print(f"=== {name} (seed {tcfg['seed']}, lr {tcfg['lr']}, epochs {tcfg['epochs']}, "
              f"patience {tcfg.get('patience_val_points')}, loss weights {loss_weights(tcfg)})")
        seed = int(tcfg.get("seed", 0))
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        data = RemixData.from_manifest(root, train_df, REPO_ROOT / "logs" / "noise_check.csv", **tcfg.get("remix", {}))
        model, how = _init_model(cfg, data)
        print(f"A. {count_params(model):,} params, {how}")
        clean_b, noisy_b = data.batch(rng, int(tcfg.get("batch_size", 16)))
        clean_t, noisy_t = torch.from_numpy(clean_b), torch.from_numpy(noisy_b)
        lw = loss_weights(tcfg)
        if name == "gru_v5b_s2":
            model.eval()  # GroupNorm only; no dropout -> deterministic either way
            with torch.no_grad():
                new, old = batch_loss(model, clean_t, noisy_t, None, lw), _old_inline_loss(model, clean_t, noisy_t, tcfg)
            same = bool(torch.equal(new, old))
            print(f"   batch_loss == exp/r2-v6 inline loss (bitwise): {same} ({new.item():.8f} vs {old.item():.8f})")
            ok_all &= same
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
        losses = []
        for step in range(3):
            loss = batch_loss(model, clean_t, noisy_t, None, lw)
            losses.append(loss.item())
            if step == 2:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 5.0)))
            opt.step()
        finite = all(np.isfinite(losses))
        net = losses[2] < losses[0]
        mono = losses[1] < losses[0] and losses[2] < losses[1]
        print(f"   fixed-batch loss before/after step 1/after step 2: "
              + " / ".join(f"{v:.5f}" for v in losses) + f"  finite {finite}, net decrease {net}, monotone {mono}")
        ok_all &= finite and net

        ck = out / name
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config", cfg_path, "--ckpt-dir", str(ck),
               "--max-steps", "2", "--val-subset", "3", "--device", "cpu", "--val-jobs", "3"]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        print("B. " + " ".join(cmd[1:]))
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        print("   " + "\n   ".join(res.stdout.strip().splitlines()[-3:]))
        if res.returncode != 0:
            print(res.stderr[-2000:])
            ok_all = False
            continue
        log = (ck / "train_log.csv").read_text(encoding="utf-8").strip().splitlines()
        print(f"   train_log.csv: {log[-1]}")

        m = load_weights(ck / "best.pt")
        meta = torch.load(ck / "best.pt", map_location="cpu", weights_only=True)
        print(f"C. best.pt weights_only=True OK: model_cfg {meta['model_cfg']}, epoch {meta['epoch']}, "
              f"pesq {meta['pesq']:.4f}, {count_params(m):,} params")
        try:
            last = torch.load(ck / "last.pt", map_location="cpu", weights_only=True)
            print(f"   last.pt weights_only=True OK (epoch {last['epoch']}, step {last['step']})")
        except Exception as e:  # last.pt is the resume state, not an inference file
            print(f"   last.pt weights_only=True: not loadable ({type(e).__name__}); weights_only=False resume only")

        params = dict(cfg["params"], weights=str(ck / "best.pt"))
        method = get_method(cfg["method"], **params)
        for _, row in val_df.iterrows():
            noisy = read_row(root, row, "noisy")
            sr = int(row["sample_rate_hz"])
            y = method(noisy, sr)
            wav = out / "infer" / name / Path(row["noisy_path"]).name
            write_wav_pcm16(wav, y, sr)
            info = sf.info(str(wav))
            good = y.shape == noisy.shape and info.samplerate == sr == 16000 and info.channels == 1 \
                and info.frames == noisy.shape[-1] and np.isfinite(y).all()
            print(f"D. {Path(row['noisy_path']).name}: in {noisy.shape[-1]} samples @ {sr} Hz -> "
                  f"out {info.frames} @ {info.samplerate} Hz, {info.channels} ch: {'OK' if good else 'FAIL'}")
            ok_all &= good
    print("SMOKE " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
