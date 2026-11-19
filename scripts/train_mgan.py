"""MetricGAN+-style fine-tuning of a GRU magnitude-mask model with a learned PESQ discriminator (exp/r2-mgan).

Two modes (config: configs/dl/gru_v6m*.yaml, section train.mgan):
  --pretrain-d PATH  train D alone for train.mgan.d_pretrain_epochs on the frozen init_from generator (gru_v5b)
                     and save {disc, opt_d, replay} to PATH (skipped if PATH already holds a finished pretrain;
                     an unfinished one restarts from scratch). One pretrained D is shared by all variants.
  (default)          G + D training. A fresh start loads G from train.init_from and D / its Adam state / the replay
                     buffer from --d-init (the pretrain file); --resume continues from <ckpt-dir>/last.pt.
Per batch (B remixed 3 s segments, the same remix stream as scripts/train.py):
  1. G forward; regression terms with exactly scripts/train.batch_loss's ops (w_cmag * cmag + w_mrstft * MR-STFT
     + w_sisdr * (-SI-SDR); a term with weight 0 is skipped), plus lambda_metric * mean (D(enhanced, clean) - 1)^2
     with D in eval mode and frozen. With lambda_metric 0 the G loss IS the regression loss (no adversarial term).
  2. PESQ-WB (pesq 0.0.4, 16 kHz, 16-bit quantised like main.py output) of enhanced and noisy vs clean is computed
     in a process pool (--pesq-jobs, default os.cpu_count()) while G back-propagates; a failed PESQ drops that row.
  3. G step (Adam, grad clip), then one D step: MSE of D on (clean, clean) -> 1, (enhanced, clean) -> its PESQ target,
     (noisy, clean) -> its PESQ target, plus replayed past (enhanced, clean) rows = replay_frac of the D batch.
     Targets are (PESQ - 1) / 3.5 clipped to [0, 1]. The enhanced rows then enter the reservoir replay buffer.
Validation every val_every epochs on the full validation split: PESQ (selection) and SNR. Checkpoints in --ckpt-dir:
best.pt (inference weights only, nrc.models.gru_mask.save_weights; loads with weights_only=True), last.pt (full state:
G, D, both optimizers, replay buffer, RNG; every epoch), train_log.csv (one row per epoch). Constant learning rates.
Early stop after patience_val_points val points without a val PESQ gain.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from _common import REPO_ROOT, load_config

from nrc.io import load_manifest, read_row, resolve_data_root, stratified_subset, to_pcm16
from nrc.metrics import pesq_wb, snr_db
from train import _atomic_save, _enhance_batch, _pesq16, loss_weights

LOG_FIELDS = ("epoch", "step", "g_loss", "g_cmag", "g_mrstft", "g_sisdr", "g_adv", "d_loss", "d_enh", "d_clean",
              "train_pesq_enh", "train_pesq_noisy", "pesq_failed", "val_pesq", "val_snr", "val_failed",
              "lr_g", "lr_d", "epoch_s", "val_s", "best_pesq", "early_stop")
PRE_FIELDS = ("epoch", "step", "d_loss", "d_enh", "d_clean", "train_pesq_enh", "train_pesq_noisy", "pesq_failed",
              "corr_d_pesq", "epoch_s")
DEFAULT_PRETRAIN = REPO_ROOT / "outputs" / "ckpt" / "mgan_d_pretrain" / "d_pretrain.pt"


def mgan_cfg(tcfg: dict) -> dict:
    m = tcfg.get("mgan", {})
    return {"lambda": float(m.get("lambda_metric", 0.0)), "lr_d": float(m.get("lr_d", 2e-4)),
            "replay_frac": float(m.get("replay_frac", 0.2)), "replay_capacity": int(m.get("replay_capacity", 512)),
            "pre_epochs": int(m.get("d_pretrain_epochs", 2))}


def g_regression(model, clean_t, noisy_t, lw: dict):
    """Magnitude-mask regression loss with the ops and summation order of train.batch_loss (bit-identical for a
    "cmag+mrstft+sisdr" config). Returns (loss, raw terms {cmag, mrstft, sisdr}, enhanced wave)."""
    import torch
    from nrc.losses import cmag_loss, mrstft_loss, sisdr_loss
    if model.mask_type != "mag" or lw["ccplx"] != 0:
        raise ValueError("train_mgan supports magnitude-mask models without the complex term only")
    with torch.no_grad():
        clean_spec = model.stft(clean_t)
    spec = model.stft(noisy_t)
    m = model.mask(spec.abs(), None)
    est_spec, est_mag = m * spec, m * spec.abs()
    cm = cmag_loss(est_mag, clean_spec.abs(), lw["c"])
    loss = lw["cmag"] * cm
    terms = {"cmag": cm.item(), "mrstft": float("nan"), "sisdr": float("nan")}
    est_wave = model.istft(est_spec, length=noisy_t.shape[-1])
    if lw["mrstft"] > 0:
        mr = mrstft_loss(est_wave, clean_t)
        loss = loss + lw["mrstft"] * mr
        terms["mrstft"] = mr.item()
    if lw["sisdr"] > 0:
        si = sisdr_loss(est_wave, clean_t)
        loss = loss + lw["sisdr"] * si
        terms["sisdr"] = si.item()
    return loss, terms, est_wave


def g_loss(model, disc, clean_t, noisy_t, lw: dict, lam: float):
    """G loss = regression + lam * mean (D(enh, clean) - 1)^2 (D must be in eval mode with frozen params).
    Returns (loss, terms incl. adv, enhanced wave, D(enhanced) detached)."""
    import torch
    loss, terms, est_wave = g_regression(model, clean_t, noisy_t, lw)
    if lam > 0:
        d_enh = disc(est_wave, clean_t)
        adv = ((d_enh - 1.0) ** 2).mean()
        loss = loss + lam * adv
    else:
        with torch.no_grad():
            d_enh = disc(est_wave, clean_t)
            adv = ((d_enh - 1.0) ** 2).mean()
    terms["adv"] = adv.item()
    return loss, terms, est_wave, d_enh.detach()


def _set_grad(module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def submit_pesq(pool, clean_b: np.ndarray, waves: np.ndarray) -> list:
    return [pool.submit(_pesq16, (clean_b[i].astype(np.float64), waves[i].astype(np.float64)))
            for i in range(len(waves))]


def d_step(disc, opt_d, replay, clean_t, enh_t, noisy_t, pesq_enh, pesq_noisy, replay_frac):
    """One D update; returns (d_loss, mean D(clean, clean) before the update). Adds valid enhanced rows to replay."""
    import torch
    from nrc.models.metric_disc import pesq_target
    dev = clean_t.device
    te = torch.from_numpy(pesq_target(pesq_enh)).float().to(dev)
    tn = torch.from_numpy(pesq_target(pesq_noisy)).float().to(dev)
    ve, vn = torch.isfinite(te), torch.isfinite(tn)
    xs, refs, ys = [clean_t, enh_t[ve], noisy_t[vn]], [clean_t, clean_t[ve], clean_t[vn]], \
        [torch.ones(clean_t.shape[0], device=dev), te[ve], tn[vn]]
    n_cur = sum(int(x.shape[0]) for x in xs)
    rep = replay.sample(round(n_cur * replay_frac / (1.0 - replay_frac)), dev) if replay_frac > 0 else None
    if rep is not None:
        xs.append(rep[0]); refs.append(rep[1]); ys.append(rep[2])
    disc.train()
    _set_grad(disc, True)
    pred = disc(torch.cat(xs), torch.cat(refs))
    loss = torch.mean((pred - torch.cat(ys)) ** 2)
    opt_d.zero_grad(set_to_none=True)
    loss.backward()
    opt_d.step()
    disc.eval()
    replay.add(enh_t[ve], clean_t[ve], te[ve])
    return loss.item(), pred[:clean_t.shape[0]].detach().mean().item()


def _rng_byte(t):
    """RNG state as the CPU uint8 tensor torch expects (torch.load with map_location="cuda" moves it to the GPU)."""
    import torch
    return torch.as_tensor(t).detach().cpu().to(torch.uint8)


def restore_torch_rng(ck: dict) -> None:
    """Restore the torch CPU / CUDA RNG states of a last.pt dict, whatever device or dtype they were loaded as.
    CUDA states are restored only when CUDA is available (skipped with a warning otherwise, e.g. a GPU checkpoint
    resumed on CPU); if the device count differs, the common devices are restored."""
    import torch
    torch.set_rng_state(_rng_byte(ck["torch_rng"]))
    cuda = ck.get("cuda_rng")
    if cuda is None:
        return
    if not torch.cuda.is_available():
        print("warning: checkpoint has CUDA RNG states but CUDA is not available; skipped", flush=True)
        return
    states = [_rng_byte(t) for t in cuda]
    n = min(len(states), torch.cuda.device_count())
    if n != len(states):
        print(f"warning: checkpoint has {len(states)} CUDA RNG states, {torch.cuda.device_count()} devices; "
              f"restoring {n}", flush=True)
    for i in range(n):
        torch.cuda.set_rng_state(states[i], device=i)


def _val_item(args) -> tuple[float, float]:
    clean, est = args
    pcm, _ = to_pcm16(est)
    y = pcm.astype(np.float64) / 32768.0
    try:
        s = snr_db(clean, y)
    except ValueError:
        s = float("nan")
    return pesq_wb(clean, y, 16000)[0], s


def validate(model, val, device, batch, pool) -> tuple[float, float, int]:
    model.eval()
    est = _enhance_batch(model, val["noisy"], device, batch)
    res = np.array(list(pool.map(_val_item, zip(val["clean"], est), chunksize=4)), float)
    model.train()
    p, s = res[:, 0], res[:, 1]
    ok = np.isfinite(p)
    return (float(p[ok].mean()) if ok.any() else float("nan")), float(np.nanmean(s)), int((~ok).sum())


def _fmt(v, nd=6):
    return "" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="default outputs/ckpt/<config name>")
    ap.add_argument("--pretrain-d", default=None, metavar="PATH", help="D-only pretraining mode; output file")
    ap.add_argument("--d-init", default=str(DEFAULT_PRETRAIN), help="pretrained D file for a fresh G+D start")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-unfinished-d", action="store_true", help="smoke only: accept a --max-steps pretrain")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke; prints every step)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--val-subset", type=int, default=None, help="stratified val subset (seed 0)")
    ap.add_argument("--pesq-jobs", type=int, default=os.cpu_count() or 1, help="PESQ processes (train + val)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = ap.parse_args(argv)

    import torch
    from nrc.data.remix import RemixData
    from nrc.models.gru_mask import GRUMask, count_params, save_weights
    from nrc.models.metric_disc import MetricDiscriminator, ReplayBuffer, pesq_target

    cfg = load_config(args.config)
    mcfg, tcfg = cfg.get("model", {}), cfg["train"]
    mg, lw = mgan_cfg(tcfg), loss_weights(tcfg)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    seed = int(tcfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pre_path = Path(args.pretrain_d) if args.pretrain_d else None
    if pre_path is not None and pre_path.is_file():
        st = torch.load(pre_path, map_location="cpu", weights_only=False)
        if st.get("finished") and args.max_steps is None:
            print(f"{pre_path}: pretrain finished ({st['epoch']} epochs); nothing to do")
            return 0

    root = resolve_data_root(args.data_root)
    t0 = time.perf_counter()
    data = RemixData.from_manifest(root, load_manifest(root, "train"), REPO_ROOT / "logs" / "noise_check.csv",
                                   **tcfg.get("remix", {}))
    print(f"data: train {len(data)} x {data.length} samples, load {time.perf_counter() - t0:.1f}s, device {device}, "
          f"pesq jobs {args.pesq_jobs}")

    model = GRUMask(**mcfg).to(device)
    disc = MetricDiscriminator(n_fft=model.n_fft, hop=model.hop).to(device)
    opt_d = torch.optim.Adam(disc.parameters(), lr=mg["lr_d"])
    replay = ReplayBuffer(mg["replay_capacity"], data.length, seed=seed + 1)
    bs = int(tcfg.get("batch_size", 16))
    steps_per_epoch = math.ceil(len(data) / bs)
    init_path = REPO_ROOT / tcfg["init_from"]

    def load_g_init():
        ck = torch.load(init_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state"])  # strict: same architecture as gru_v5b
        print(f"G init from {init_path.name} (epoch {ck.get('epoch')}, pesq {ck.get('pesq')}); "
              f"G {count_params(model):,} params, D {count_params(disc):,} params")

    # ---------------- D-only pretraining on the frozen init generator ----------------
    if pre_path is not None:
        load_g_init()
        model.eval()
        _set_grad(model, False)
        disc.eval()
        pre_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = pre_path.with_name(pre_path.stem + "_log.csv")
        step = 0
        with ProcessPoolExecutor(max_workers=max(1, args.pesq_jobs)) as pool, \
                open(log_path, "w", newline="", encoding="utf-8") as logf:
            log = csv.DictWriter(logf, fieldnames=PRE_FIELDS)
            log.writeheader()
            for epoch in range(mg["pre_epochs"]):
                te, acc, stop = time.perf_counter(), {k: [] for k in ("d", "de", "dc", "pe", "pn")}, False
                n_fail = 0
                for _ in range(steps_per_epoch):
                    clean_b, noisy_b = data.batch(rng, bs)
                    clean_t, noisy_t = torch.from_numpy(clean_b).to(device), torch.from_numpy(noisy_b).to(device)
                    with torch.no_grad():
                        _, _, est = g_regression(model, clean_t, noisy_t, lw)
                        d_enh = disc(est, clean_t)
                    fe = submit_pesq(pool, clean_b, est.cpu().double().numpy())
                    fn = submit_pesq(pool, clean_b, noisy_b)
                    pe, pn = np.array([f.result() for f in fe]), np.array([f.result() for f in fn])
                    n_fail += int((~np.isfinite(pe)).sum() + (~np.isfinite(pn)).sum())
                    dl, dc = d_step(disc, opt_d, replay, clean_t, est, noisy_t, pe, pn, mg["replay_frac"])
                    acc["d"].append(dl); acc["dc"].append(dc); acc["de"] += d_enh.cpu().tolist()
                    acc["pe"] += pe.tolist(); acc["pn"] += pn.tolist()
                    step += 1
                    if args.max_steps is not None:
                        print(f"  pretrain step {step}: d_loss {dl:.6f} D(enh) {d_enh.mean().item():.4f} "
                              f"pesq enh {np.nanmean(pe):.3f} noisy {np.nanmean(pn):.3f}", flush=True)
                        if step >= args.max_steps:
                            stop = True
                            break
                de, pe_all = np.array(acc["de"]), np.array(acc["pe"])
                ok = np.isfinite(pe_all)
                corr = float(np.corrcoef(de[ok], pesq_target(pe_all[ok]))[0, 1]) if ok.sum() > 2 else float("nan")
                row = {"epoch": epoch + 1, "step": step, "d_loss": _fmt(np.mean(acc["d"])), "d_enh": _fmt(de.mean(), 4),
                       "d_clean": _fmt(np.mean(acc["dc"]), 4), "train_pesq_enh": _fmt(np.nanmean(pe_all), 4),
                       "train_pesq_noisy": _fmt(np.nanmean(acc["pn"]), 4), "pesq_failed": n_fail,
                       "corr_d_pesq": _fmt(corr, 4), "epoch_s": f"{time.perf_counter() - te:.1f}"}
                log.writerow(row)
                logf.flush()
                print("pretrain " + " ".join(f"{k} {v}" for k, v in row.items()), flush=True)
                finished = epoch + 1 >= mg["pre_epochs"] and not stop
                _atomic_save({"disc": disc.state_dict(), "opt_d": opt_d.state_dict(), "replay": replay.state(),
                              "epoch": epoch + 1, "step": step, "finished": finished, "cfg": cfg}, pre_path)
                if stop:
                    break
        print(f"done: D pretrain -> {pre_path}")
        return 0

    # ---------------- G + D training ----------------
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else REPO_ROOT / "outputs" / "ckpt" / cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path, log_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt", ckpt_dir / "train_log.csv"
    opt_g = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 1e-4)))
    start_epoch, step, best, best_epoch, since_best = 0, 0, -math.inf, -1, 0
    if args.resume and last_path.is_file():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"]); replay.load(ck["replay"])
        start_epoch, step, best, best_epoch = ck["epoch"], ck["step"], ck["best_pesq"], ck["best_epoch"]
        since_best = int(ck.get("since_best", 0))
        if ck.get("early_stopped"):
            print(f"{last_path} is early-stopped (epoch {start_epoch}, best {best:.4f} ep {best_epoch}); nothing to do")
            return 0
        rng.bit_generator.state = ck["np_rng"]
        restore_torch_rng(ck)
        print(f"resumed from {last_path}: epoch {start_epoch}, step {step}, best {best:.4f} (ep {best_epoch}), "
              f"replay {len(replay)}")
    else:
        load_g_init()
        d_init = Path(args.d_init)
        if not d_init.is_file():
            raise FileNotFoundError(f"pretrained D {d_init} missing: run --pretrain-d first")
        st = torch.load(d_init, map_location=device, weights_only=False)
        if not st.get("finished") and not args.allow_unfinished_d:
            raise RuntimeError(f"{d_init} is an unfinished pretrain (epoch {st.get('epoch')})")
        disc.load_state_dict(st["disc"]); opt_d.load_state_dict(st["opt_d"]); replay.load(st["replay"])
        for g in opt_d.param_groups:
            g["lr"] = mg["lr_d"]
        print(f"D init from {d_init} (pretrain epochs {st['epoch']}, replay {len(replay)})")
    lam, clip = mg["lambda"], float(tcfg.get("grad_clip", 5.0))
    epochs, val_every = int(tcfg["epochs"]), int(tcfg.get("val_every", 1))
    patience = int(tcfg["patience_val_points"]) if tcfg.get("patience_val_points") else None
    print(f"G loss = {lw['cmag']} cmag + {lw['mrstft']} MR-STFT + {lw['sisdr']} (-SI-SDR) + {lam} (D - 1)^2; "
          f"lr G {opt_g.param_groups[0]['lr']:.1e} D {mg['lr_d']:.1e}; {epochs} epochs x {steps_per_epoch} steps")
    val_df = stratified_subset(load_manifest(root, "validation"), args.val_subset)
    val = {k: np.stack([read_row(root, r, k).astype(np.float32 if k == "noisy" else np.float64)
                        for _, r in val_df.iterrows()]) for k in ("clean", "noisy")}
    new_log = not log_path.is_file()
    model.train()
    disc.eval()
    with ProcessPoolExecutor(max_workers=max(1, args.pesq_jobs)) as pool, \
            open(log_path, "a", newline="", encoding="utf-8") as logf:
        log = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        if new_log:
            log.writeheader()
        for epoch in range(start_epoch, epochs):
            te, stop, early, n_fail = time.perf_counter(), False, False, 0
            acc = {k: [] for k in ("g", "cmag", "mrstft", "sisdr", "adv", "d", "dc", "de", "pe", "pn")}
            for _ in range(steps_per_epoch):
                clean_b, noisy_b = data.batch(rng, bs)
                clean_t, noisy_t = torch.from_numpy(clean_b).to(device), torch.from_numpy(noisy_b).to(device)
                t_a = time.perf_counter()
                _set_grad(disc, False)
                loss, terms, est, d_enh = g_loss(model, disc, clean_t, noisy_t, lw, lam)
                est_d = est.detach()
                fe = submit_pesq(pool, clean_b, est_d.cpu().double().numpy())  # overlaps the G backward pass
                fn = submit_pesq(pool, clean_b, noisy_b)
                opt_g.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt_g.step()
                t_b = time.perf_counter()
                pe, pn = np.array([f.result() for f in fe]), np.array([f.result() for f in fn])
                n_fail += int((~np.isfinite(pe)).sum() + (~np.isfinite(pn)).sum())
                t_c = time.perf_counter()
                dl, dc = d_step(disc, opt_d, replay, clean_t, est_d, noisy_t, pe, pn, mg["replay_frac"])
                acc["g"].append(loss.item()); acc["d"].append(dl); acc["dc"].append(dc)
                for k in ("cmag", "mrstft", "sisdr", "adv"):
                    acc[k].append(terms[k])
                acc["de"] += d_enh.cpu().tolist(); acc["pe"] += pe.tolist(); acc["pn"] += pn.tolist()
                step += 1
                if args.max_steps is not None:
                    print(f"  step {step}: g_loss {loss.item():.6f} (cmag {terms['cmag']:.5f} mrstft {terms['mrstft']:.5f} "
                          f"sisdr {terms['sisdr']:.4f} adv {terms['adv']:.5f}) d_loss {dl:.6f} "
                          f"D(enh) {d_enh.mean().item():.4f} pesq enh {np.nanmean(pe):.3f} noisy {np.nanmean(pn):.3f} "
                          f"[G {t_b - t_a:.3f}s, PESQ wait {t_c - t_b:.3f}s, D {time.perf_counter() - t_c:.3f}s]",
                          flush=True)
                    if step >= args.max_steps:
                        stop = True
                        break
            epoch_s = time.perf_counter() - te
            pesq_v, snr_v, failed, val_s = float("nan"), float("nan"), "", 0.0
            if (epoch + 1) % val_every == 0 or epoch + 1 == epochs or stop:
                tv = time.perf_counter()
                pesq_v, snr_v, failed = validate(model, val, device, int(tcfg.get("val_batch", 16)), pool)
                val_s = time.perf_counter() - tv
                if math.isfinite(pesq_v):
                    if pesq_v > best:
                        best, best_epoch, since_best = pesq_v, epoch + 1, 0
                        tmp = best_path.with_suffix(".pt.tmp")
                        save_weights(tmp, model, epoch=best_epoch, pesq=best, config=cfg["name"])
                        os.replace(tmp, best_path)
                    else:
                        since_best += 1
                    early = patience is not None and since_best >= patience
            _atomic_save({"model": model.state_dict(), "disc": disc.state_dict(), "opt_g": opt_g.state_dict(),
                          "opt_d": opt_d.state_dict(), "replay": replay.state(), "epoch": epoch + 1, "step": step,
                          "best_pesq": best, "best_epoch": best_epoch, "since_best": since_best,
                          "early_stopped": early, "np_rng": rng.bit_generator.state,
                          "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                          "cfg": cfg}, last_path)
            row = {"epoch": epoch + 1, "step": step, "g_loss": _fmt(np.mean(acc["g"])),
                   "g_cmag": _fmt(np.mean(acc["cmag"])), "g_mrstft": _fmt(np.mean(acc["mrstft"])),
                   "g_sisdr": _fmt(np.mean(acc["sisdr"]), 4), "g_adv": _fmt(np.mean(acc["adv"])),
                   "d_loss": _fmt(np.mean(acc["d"])), "d_enh": _fmt(np.mean(acc["de"]), 4),
                   "d_clean": _fmt(np.mean(acc["dc"]), 4), "train_pesq_enh": _fmt(np.nanmean(acc["pe"]), 4),
                   "train_pesq_noisy": _fmt(np.nanmean(acc["pn"]), 4), "pesq_failed": n_fail,
                   "val_pesq": _fmt(pesq_v, 4), "val_snr": _fmt(snr_v, 3), "val_failed": failed,
                   "lr_g": f"{opt_g.param_groups[0]['lr']:.2e}", "lr_d": f"{opt_d.param_groups[0]['lr']:.2e}",
                   "epoch_s": f"{epoch_s:.1f}", "val_s": f"{val_s:.1f}",
                   "best_pesq": f"{best:.4f}" if best > -math.inf else "", "early_stop": "1" if early else ""}
            log.writerow(row)
            logf.flush()
            print(f"epoch {epoch + 1}/{epochs} step {step} g {row['g_loss']} adv {row['g_adv']} d {row['d_loss']} "
                  f"D(enh) {row['d_enh']} pesq_tr {row['train_pesq_enh']} val_pesq {row['val_pesq']} "
                  f"val_snr {row['val_snr']} best {best:.4f} (ep {best_epoch}) "
                  f"[{epoch_s:.1f}s + val {val_s:.1f}s, {epoch_s / max(1, len(acc['g'])):.3f}s/step]", flush=True)
            if early:
                print(f"early stop: no val PESQ gain for {since_best} val points", flush=True)
            if stop or early:
                break
    print(f"done: best val PESQ {best:.4f} at epoch {best_epoch}; weights {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
