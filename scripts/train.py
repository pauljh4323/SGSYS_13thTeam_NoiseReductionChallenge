"""Train the BiGRU mask model (configs/dl/*.yaml) on the train split with remix augmentation.

Validation PESQ (wideband, 16-bit quantized exactly like main.py writes) runs on CPU workers every
`val_every` epochs. Checkpoints in --ckpt-dir: last.pt (full training state, every epoch),
best.pt (inference weights, on val PESQ gain), train_log.csv (one row per epoch).
--resume continues from last.pt if present (fresh start otherwise), so it is safe to always pass.
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
from nrc.metrics import pesq_wb

LOG_FIELDS = ("epoch", "step", "train_loss", "val_pesq", "val_failed", "lr", "epoch_s", "val_s", "best_pesq")


def _atomic_save(obj, path: Path) -> None:
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _norm_stats(model, noisy: np.ndarray, device, chunk: int = 64):
    """Per-bin mean/std of log1p|X| over the train noisy set."""
    import torch
    s1 = s2 = None
    count = 0
    with torch.no_grad():
        for i in range(0, len(noisy), chunk):
            x = torch.from_numpy(noisy[i:i + chunk]).to(device)
            f = torch.log1p(model.stft(x).abs())  # (B, F, T)
            a, b = f.sum(dim=(0, 2)), (f ** 2).sum(dim=(0, 2))
            s1, s2 = (a, b) if s1 is None else (s1 + a, s2 + b)
            count += f.shape[0] * f.shape[2]
    mean = s1 / count
    return mean, (s2 / count - mean ** 2).clamp_min(0).sqrt()


def _enhance_batch(model, noisy: np.ndarray, device, batch: int) -> np.ndarray:
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(noisy), batch):
            out.append(model(torch.from_numpy(noisy[i:i + batch]).to(device)).cpu().double().numpy())
    return np.concatenate(out)


def _pesq16(args) -> float:
    clean, est = args
    pcm, _ = to_pcm16(est)
    return pesq_wb(clean, pcm.astype(np.float64) / 32768.0, 16000)[0]


def validate(model, val, device, batch, pool) -> tuple[float, int]:
    model.eval()
    est = _enhance_batch(model, val["noisy"], device, batch)
    scores = np.array(list(pool.map(_pesq16, zip(val["clean"], est), chunksize=4)), float)
    model.train()
    ok = np.isfinite(scores)
    return (float(scores[ok].mean()) if ok.any() else float("nan")), int((~ok).sum())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ckpt-dir", default=None, help="default outputs/ckpt/<config name>")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke)")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--val-subset", type=int, default=None, help="stratified val subset (seed 0)")
    parser.add_argument("--val-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = parser.parse_args(argv)

    import torch
    from nrc.data.remix import RemixData
    from nrc.models.gru_mask import GRUMask, count_params, save_weights

    cfg = load_config(args.config)
    mcfg, tcfg = cfg.get("model", {}), cfg["train"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else REPO_ROOT / "outputs" / "ckpt" / cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed = int(tcfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    root = resolve_data_root(args.data_root)
    t0 = time.perf_counter()
    train_df = load_manifest(root, "train")
    data = RemixData.from_manifest(root, train_df, REPO_ROOT / "logs" / "noise_check.csv",
                                   **tcfg.get("remix", {}))
    val_df = stratified_subset(load_manifest(root, "validation"), args.val_subset)
    val = {k: np.stack([read_row(root, r, k).astype(np.float32 if k == "noisy" else np.float64)
                        for _, r in val_df.iterrows()]) for k in ("clean", "noisy")}
    print(f"data: train {len(data)} (noise sources {data.noise_idx.size}), val {len(val_df)}, "
          f"load {time.perf_counter() - t0:.1f}s, device {device}")

    model = GRUMask(**mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 1e-3)))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=float(tcfg.get("plateau_factor", 0.5)),
        patience=int(tcfg.get("plateau_patience", 3)), min_lr=float(tcfg.get("min_lr", 1e-5)))
    start_epoch, step, best, best_epoch = 0, 0, -math.inf, -1
    last_path, best_path, log_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt", ckpt_dir / "train_log.csv"
    if args.resume and last_path.is_file():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, step, best, best_epoch = ck["epoch"], ck["step"], ck["best_pesq"], ck["best_epoch"]
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if device == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"resumed from {last_path}: epoch {start_epoch}, step {step}, best {best:.4f} (ep {best_epoch})")
    else:
        model.set_norm_stats(*_norm_stats(model, data.noisy, device))
    print(f"model: {count_params(model):,} params")

    epochs, bs = int(tcfg["epochs"]), int(tcfg.get("batch_size", 16))
    steps_per_epoch = math.ceil(len(data) / bs)
    val_every, c = int(tcfg.get("val_every", 1)), float(tcfg.get("compress", 0.3))
    clip = float(tcfg.get("grad_clip", 5.0))
    new_log = not log_path.is_file()
    model.train()
    with ProcessPoolExecutor(max_workers=max(1, args.val_jobs)) as pool, \
            open(log_path, "a", newline="", encoding="utf-8") as logf:
        log = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        if new_log:
            log.writeheader()
        for epoch in range(start_epoch, epochs):
            te, losses, stop = time.perf_counter(), [], False
            for _ in range(steps_per_epoch):
                clean_b, noisy_b = data.batch(rng, bs)
                clean_t, noisy_t = torch.from_numpy(clean_b).to(device), torch.from_numpy(noisy_b).to(device)
                with torch.no_grad():
                    target = model.stft(clean_t).abs()
                mag = model.stft(noisy_t).abs()
                est = model.mask(mag) * mag
                loss = torch.mean((est.clamp_min(1e-8) ** c - target.clamp_min(1e-8) ** c) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                losses.append(loss.item())
                step += 1
                if args.max_steps is not None and step >= args.max_steps:
                    stop = True
                    break
            epoch_s = time.perf_counter() - te
            pesq_v, failed, val_s = float("nan"), "", 0.0
            if (epoch + 1) % val_every == 0 or epoch + 1 == epochs or stop:
                tv = time.perf_counter()
                pesq_v, failed = validate(model, val, device, int(tcfg.get("val_batch", 16)), pool)
                val_s = time.perf_counter() - tv
                if math.isfinite(pesq_v):
                    sched.step(pesq_v)
                    if pesq_v > best:
                        best, best_epoch = pesq_v, epoch + 1
                        tmp = best_path.with_suffix(".pt.tmp")
                        save_weights(tmp, model, epoch=best_epoch, pesq=best, config=cfg["name"])
                        os.replace(tmp, best_path)
            lr = opt.param_groups[0]["lr"]
            _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                          "scheduler": sched.state_dict(), "epoch": epoch + 1, "step": step,
                          "best_pesq": best, "best_epoch": best_epoch, "np_rng": rng.bit_generator.state,
                          "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                          "cfg": cfg}, last_path)
            log.writerow({"epoch": epoch + 1, "step": step, "train_loss": f"{np.mean(losses):.6f}",
                          "val_pesq": "" if not math.isfinite(pesq_v) else f"{pesq_v:.4f}",
                          "val_failed": failed, "lr": f"{lr:.2e}", "epoch_s": f"{epoch_s:.1f}",
                          "val_s": f"{val_s:.1f}", "best_pesq": f"{best:.4f}" if best > -math.inf else ""})
            logf.flush()
            print(f"epoch {epoch + 1}/{epochs} step {step} loss {np.mean(losses):.5f} "
                  f"val_pesq {pesq_v:.4f} best {best:.4f} (ep {best_epoch}) lr {lr:.1e} "
                  f"[{epoch_s:.1f}s + val {val_s:.1f}s, {epoch_s / max(1, len(losses)):.3f}s/step]", flush=True)
            if stop:
                break
    print(f"done: best val PESQ {best:.4f} at epoch {best_epoch}; weights {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
