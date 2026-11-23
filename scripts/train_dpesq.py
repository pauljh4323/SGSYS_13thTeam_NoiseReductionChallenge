"""Fine-tune a GRU magnitude-mask model with a differentiable PESQ loss (torch-pesq via nrc.tpesq; exp/r2-dpesq).

Config: configs/dl/gru_v8p*.yaml. Warm start from train.init_from (gru_v5b), the same remix batch stream as
scripts/train.py / train_mgan.py (data.batch(rng, B) per step), then per batch:
  loss = w_cmag * cmag + w_mrstft * MR-STFT + w_sisdr * (-SI-SDR)          (train_mgan.g_regression, train.batch_loss ops)
       + lambda_pesq * mean_b [0.1 d_symm + 0.0309 d_asymm](clean_b, est_b)  (= torch_pesq.PesqLoss(1.0).forward)
The PESQ term is skipped when lambda_pesq == 0, so the loss is then bit-identical to train.batch_loss.
Rows whose clean or estimate is digital silence (max|x| < 1e-8) are dropped from the PESQ term (0/0 in the level
alignment -> NaN, logs/r2_dpesq_proxy.log); a non-finite PESQ term skips that term for the step (counted; > 5 per
epoch aborts). Adam, constant lr, grad clip. Every val_every epochs: val PESQ (pesq 0.0.4 on 16-bit output, selection)
+ SNR on the full validation split, and train64_pesq on a fixed 64-file stratified train subset (seed 0, original
mixtures) so the train/val gap is visible. Checkpoints in --ckpt-dir: best.pt (inference weights only, loads with
weights_only=True), last.pt (full state incl. RNG, every epoch; resume via train_mgan.restore_torch_rng),
train_log.csv (one row per epoch). Early stop after patience_val_points val points without a val PESQ gain.
exp/r2-dmix: train.rms_norm_target (gru_v8p2z1/z2) turns on train-time rms_norm in RemixData (gain augmentation off)
and makes validation + train64 match inference level_mode "rms_norm": each noisy file is scaled by
g = target / max(rms, RMS_EPS) (float64, then float32) and each output is scaled down as a whole if its peak > 0.99.
The config's inference params must then be level_mode rms_norm with the same target_rms. train.remix.p_dm > 0 turns
on dynamic mixing (nrc.data.remix). Both unset: unchanged behaviour (bit-identical loss stream).
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

from nrc.io import load_manifest, read_row, resolve_data_root, stratified_subset
from train import _atomic_save, _enhance_batch, loss_weights
from train_mgan import _val_item, g_regression, restore_torch_rng, validate

from nrc.methods.gru_mask import OUT_PEAK, RMS_EPS

LOG_FIELDS = ("epoch", "step", "loss", "l_cmag", "l_mrstft", "l_sisdr", "l_pesq", "tpesq_mos", "pesq_rows_dropped",
              "pesq_nonfinite", "train64_pesq", "val_pesq", "val_snr", "gap", "val_failed", "lr", "epoch_s", "val_s",
              "best_pesq", "early_stop")
SILENT = 1e-8
MAX_NONFINITE = 5
TRAIN_SUBSET = 64


def pesq_term(loss_fn, clean_t, est_wave):
    """(PESQ loss over non-silent rows (scalar tensor or None), detached MOS of those rows (np), rows dropped).
    loss = mean_b (0.1 d_symm + 0.0309 d_asymm) = PesqLoss(1.0).forward(clean, est).mean(); MOS uses the same raw
    distances as PesqLoss.mos."""
    import torch
    with torch.no_grad():
        keep = (clean_t.abs().amax(-1) >= SILENT) & (est_wave.abs().amax(-1) >= SILENT)
    dropped = int((~keep).sum())
    if not bool(keep.any()):
        return None, np.array([]), dropped
    if dropped:
        clean_t, est_wave = clean_t[keep], est_wave[keep]
    d_symm, d_asymm = loss_fn.raw(clean_t, est_wave)
    per = 0.1 * d_symm + 0.0309 * d_asymm
    with torch.no_grad():
        mos = 0.999 + 4 / (1 + torch.exp(-1.3669 * (4.5 - per) + 3.8224))
    return per.mean(), mos.detach().cpu().double().numpy(), dropped


def dpesq_loss(model, loss_fn, clean_t, noisy_t, lw: dict, lam: float, need_mos: bool = True):
    """Total loss, raw terms, MOS of the batch (np), rows dropped, PESQ term finite?"""
    import torch
    loss, terms, est = g_regression(model, clean_t, noisy_t, lw)
    terms["pesq"], mos, dropped, finite = float("nan"), np.array([]), 0, True
    if lam > 0:
        pl, mos, dropped = pesq_term(loss_fn, clean_t, est)
        if pl is not None:
            finite = bool(torch.isfinite(pl))
            if finite:
                loss = loss + lam * pl
                terms["pesq"] = pl.item()
    elif need_mos:
        with torch.no_grad():
            pl, mos, dropped = pesq_term(loss_fn, clean_t, est.detach())
            terms["pesq"] = float("nan") if pl is None else pl.item()
    return loss, terms, mos, dropped, finite


def grad_norm(model, loss) -> float:
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads if g is not None)))


def calibrate(model, loss_fn, data, rng, lw_ref: dict, bs: int, n_batches: int, device) -> dict:
    """Ratio ||grad(regression loss with lw_ref)|| / ||grad(PESQ term)|| at the current weights, per batch."""
    import torch
    reg, pq = [], []
    for _ in range(n_batches):
        clean_b, noisy_b = data.batch(rng, bs)
        clean_t, noisy_t = torch.from_numpy(clean_b).to(device), torch.from_numpy(noisy_b).to(device)
        loss, _, est = g_regression(model, clean_t, noisy_t, lw_ref)
        pl, _, _ = pesq_term(loss_fn, clean_t, est)
        reg.append(grad_norm(model, loss))
        pq.append(grad_norm(model, pl))
    reg, pq = np.array(reg), np.array(pq)
    return {"reg": reg, "pesq": pq, "ratio": reg / pq, "lambda": float(np.mean(reg / pq))}


def rms_norm_rows(noisy64: np.ndarray, target: float) -> np.ndarray:
    """Per file, as methods.gru_mask.enhance (level_mode rms_norm): (x * target / max(rms(x), RMS_EPS)) -> float32."""
    out = np.empty(noisy64.shape, np.float32)
    for r, x in enumerate(noisy64):
        rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
        out[r] = (x * (float(target) / max(rms, RMS_EPS))).astype(np.float32)
    return out


def enhance_rmsn(model, noisy_n: np.ndarray, device, batch) -> np.ndarray:
    """Model on rms-normalised inputs (rms_norm_rows); each output scaled down as a whole if its peak > OUT_PEAK."""
    model.eval()
    est = _enhance_batch(model, noisy_n, device, batch)
    model.train()
    pk = np.abs(est).max(axis=1)
    for r in np.flatnonzero(pk > OUT_PEAK):
        est[r] = est[r] * (OUT_PEAK / pk[r])
    return est


def validate_rmsn(model, val, device, batch, pool) -> tuple[float, float, int]:
    """train_mgan.validate on rms-normalised inputs with the inference output peak cap (= level_mode rms_norm)."""
    est = enhance_rmsn(model, val["noisy"], device, batch)
    res = np.array(list(pool.map(_val_item, zip(val["clean"], est), chunksize=4)), float)
    p, s = res[:, 0], res[:, 1]
    ok = np.isfinite(p)
    return (float(p[ok].mean()) if ok.any() else float("nan")), float(np.nanmean(s)), int((~ok).sum())


def rms_target(cfg: dict) -> float | None:
    """train.rms_norm_target, checked against the inference params (level_mode rms_norm, same target_rms)."""
    t = cfg["train"].get("rms_norm_target")
    if t is None:
        return None
    t, pr = float(t), cfg.get("params", {})
    if not (pr.get("level_norm") and pr.get("level_mode") == "rms_norm" and float(pr.get("target_rms", -1)) == t):
        raise ValueError(f"train.rms_norm_target {t} needs params level_norm true, level_mode rms_norm, target_rms {t}")
    return t


def _fmt(v, nd=6):
    return "" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="default outputs/ckpt/<config name>")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke; prints every step)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--val-subset", type=int, default=None, help="stratified val subset (seed 0)")
    ap.add_argument("--train-subset", type=int, default=TRAIN_SUBSET, help="train64 PESQ subset size")
    ap.add_argument("--pesq-jobs", type=int, default=os.cpu_count() or 1, help="PESQ processes (val + train64)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = ap.parse_args(argv)

    import torch
    from nrc.data.remix import RemixData
    from nrc.models.gru_mask import GRUMask, count_params, save_weights
    from nrc.tpesq import pesq_loss

    cfg = load_config(args.config)
    mcfg, tcfg = cfg.get("model", {}), cfg["train"]
    lw, lam = loss_weights(tcfg), float(tcfg.get("lambda_pesq", 0.0))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    seed = int(tcfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    root = resolve_data_root(args.data_root)
    t0 = time.perf_counter()
    train_df = load_manifest(root, "train")
    rms_t = rms_target(cfg)
    remix_kw = dict(tcfg.get("remix", {}))
    if rms_t is not None:
        remix_kw["rms_norm_target"] = rms_t
    data = RemixData.from_manifest(root, train_df, REPO_ROOT / "logs" / "noise_check.csv", **remix_kw)
    run_val = validate if rms_t is None else validate_rmsn

    def load_set(df):
        if rms_t is not None:
            d = {k: np.stack([read_row(root, r, k).astype(np.float64) for _, r in df.iterrows()])
                 for k in ("clean", "noisy")}
            d["noisy"] = rms_norm_rows(d["noisy"], rms_t)
            return d
        return {k: np.stack([read_row(root, r, k).astype(np.float32 if k == "noisy" else np.float64)
                             for _, r in df.iterrows()]) for k in ("clean", "noisy")}
    val = load_set(stratified_subset(load_manifest(root, "validation"), args.val_subset))
    tr64 = load_set(stratified_subset(train_df, args.train_subset))
    print(f"data: train {len(data)} x {data.length} samples, val {len(val['clean'])}, train subset "
          f"{len(tr64['clean'])}, load {time.perf_counter() - t0:.1f}s, device {device}, pesq jobs {args.pesq_jobs}")
    print(f"data options: rms_norm_target {rms_t} (train + val/train64), gain augmentation "
          f"{'off' if rms_t is not None else data.gain_mode + ' ' + str(data.gain_db)}, dynamic mixing p_dm {data.p_dm}"
          + (f" (snr {data.dm_snr_db} dB, p_second {data.dm_p_second}, rel {data.dm_second_rel_db} dB)"
             if data.p_dm > 0 else ""), flush=True)

    model = GRUMask(**mcfg).to(device)
    if model.mask_type != "mag":
        raise ValueError("train_dpesq supports magnitude-mask models only")
    loss_fn = pesq_loss(1.0, 16000).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 5e-5)))
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else REPO_ROOT / "outputs" / "ckpt" / cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path, log_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt", ckpt_dir / "train_log.csv"
    start_epoch, step, best, best_epoch, since_best = 0, 0, -math.inf, -1, 0
    if args.resume and last_path.is_file():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch, step, best, best_epoch = ck["epoch"], ck["step"], ck["best_pesq"], ck["best_epoch"]
        since_best = int(ck.get("since_best", 0))
        if ck.get("early_stopped"):
            print(f"{last_path} is early-stopped (epoch {start_epoch}, best {best:.4f} ep {best_epoch}); nothing to do")
            return 0
        rng.bit_generator.state = ck["np_rng"]
        restore_torch_rng(ck)
        print(f"resumed from {last_path}: epoch {start_epoch}, step {step}, best {best:.4f} (ep {best_epoch})")
    else:
        init_path = REPO_ROOT / tcfg["init_from"]
        ck = torch.load(init_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state"])  # strict: same architecture as gru_v5b
        print(f"init from {init_path.name} (epoch {ck.get('epoch')}, pesq {ck.get('pesq')}); "
              f"{count_params(model):,} params; fresh optimizer")

    bs = int(tcfg.get("batch_size", 16))
    steps_per_epoch = math.ceil(len(data) / bs)
    clip, epochs, val_every = float(tcfg.get("grad_clip", 5.0)), int(tcfg["epochs"]), int(tcfg.get("val_every", 1))
    patience = int(tcfg["patience_val_points"]) if tcfg.get("patience_val_points") else None
    print(f"loss = {lw['cmag']} cmag + {lw['mrstft']} MR-STFT + {lw['sisdr']} (-SI-SDR) + {lam} PESQ; "
          f"lr {opt.param_groups[0]['lr']:.1e} (constant); {epochs} epochs x {steps_per_epoch} steps", flush=True)
    new_log = not log_path.is_file()
    model.train()
    vb = int(tcfg.get("val_batch", 16))
    with ProcessPoolExecutor(max_workers=max(1, args.pesq_jobs)) as pool, \
            open(log_path, "a", newline="", encoding="utf-8") as logf:
        log = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        if new_log:
            log.writeheader()
        for epoch in range(start_epoch, epochs):
            te, stop, early = time.perf_counter(), False, False
            acc = {k: [] for k in ("loss", "cmag", "mrstft", "sisdr", "pesq", "mos")}
            dropped_n, nonfinite_n = 0, 0
            for _ in range(steps_per_epoch):
                clean_b, noisy_b = data.batch(rng, bs)
                clean_t, noisy_t = torch.from_numpy(clean_b).to(device), torch.from_numpy(noisy_b).to(device)
                ts = time.perf_counter()
                loss, terms, mos, dropped, finite = dpesq_loss(model, loss_fn, clean_t, noisy_t, lw, lam)
                if not finite:
                    nonfinite_n += 1
                    print(f"warning: non-finite PESQ term at step {step + 1}; term skipped", flush=True)
                    if nonfinite_n > MAX_NONFINITE:
                        raise RuntimeError(f"more than {MAX_NONFINITE} non-finite PESQ terms in epoch {epoch + 1}")
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step + 1}: {terms}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                dropped_n += dropped
                acc["loss"].append(loss.item())
                for k in ("cmag", "mrstft", "sisdr", "pesq"):
                    acc[k].append(terms[k])
                acc["mos"] += mos.tolist()
                step += 1
                if args.max_steps is not None:
                    print(f"  step {step}: loss {loss.item():.6f} (cmag {terms['cmag']:.5f} mrstft {terms['mrstft']:.5f} "
                          f"sisdr {terms['sisdr']:.4f} pesq {terms['pesq']:.5f}) tpesq_mos {np.mean(mos):.4f} "
                          f"dropped {dropped} [{time.perf_counter() - ts:.3f}s]", flush=True)
                    if step >= args.max_steps:
                        stop = True
                        break
            epoch_s = time.perf_counter() - te
            pesq_v = snr_v = tr_v = float("nan")
            failed, val_s = "", 0.0
            if (epoch + 1) % val_every == 0 or epoch + 1 == epochs or stop:
                tv = time.perf_counter()
                pesq_v, snr_v, failed = run_val(model, val, device, vb, pool)
                tr_v, _, _ = run_val(model, tr64, device, vb, pool)
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
            _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch + 1,
                          "step": step, "best_pesq": best, "best_epoch": best_epoch, "since_best": since_best,
                          "early_stopped": early, "np_rng": rng.bit_generator.state,
                          "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                          "cfg": cfg}, last_path)
            mean = {k: (float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else float("nan"))
                    for k, v in acc.items()}
            row = {"epoch": epoch + 1, "step": step, "loss": _fmt(mean["loss"]), "l_cmag": _fmt(mean["cmag"]),
                   "l_mrstft": _fmt(mean["mrstft"]), "l_sisdr": _fmt(mean["sisdr"]), "l_pesq": _fmt(mean["pesq"]),
                   "tpesq_mos": _fmt(mean["mos"], 4), "pesq_rows_dropped": dropped_n, "pesq_nonfinite": nonfinite_n,
                   "train64_pesq": _fmt(tr_v, 4), "val_pesq": _fmt(pesq_v, 4), "val_snr": _fmt(snr_v, 3),
                   "gap": _fmt(tr_v - pesq_v, 4), "val_failed": failed, "lr": f"{opt.param_groups[0]['lr']:.2e}",
                   "epoch_s": f"{epoch_s:.1f}", "val_s": f"{val_s:.1f}",
                   "best_pesq": f"{best:.4f}" if best > -math.inf else "", "early_stop": "1" if early else ""}
            log.writerow(row)
            logf.flush()
            print(f"epoch {epoch + 1}/{epochs} step {step} loss {row['loss']} l_pesq {row['l_pesq']} "
                  f"tpesq_mos {row['tpesq_mos']} train64 {row['train64_pesq']} val_pesq {row['val_pesq']} "
                  f"(gap {row['gap']}) val_snr {row['val_snr']} best {best:.4f} (ep {best_epoch}) "
                  f"[{epoch_s:.1f}s + val {val_s:.1f}s, {epoch_s / max(1, len(acc['loss'])):.3f}s/step]", flush=True)
            if early:
                print(f"early stop: no val PESQ gain for {since_best} val points", flush=True)
            if stop or early:
                break
    print(f"done: best val PESQ {best:.4f} at epoch {best_epoch}; weights {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
