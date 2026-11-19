"""exp/r2-dpesq proxy check: does torch-pesq (nrc.tpesq, PesqLoss(1.0, 16 kHz).mos) track pesq 0.0.4 WB on our outputs?

Per file of validation (164) + val_remix (328): gru_v5b (configs/dl/gru_v5b_off.yaml, enhanced in memory and 16-bit
quantised exactly like main.py / eval.py), noisy, and dsp-v8 (val_remix only, cached outputs/remix_pred_dsp_v8).
Reports Pearson / Spearman r of torch-pesq MOS vs pesq 0.0.4, per file and on paired differences (v5b - noisy,
v5b - dsp-v8), checks the shim's lfilter against scipy, and the PesqLoss gradient on silence and normal 3 s inputs.
Gate (exp/r2-dpesq): paired-difference Spearman (v5b - noisy, all 492 files) >= 0.6, else no training.
Writes logs/r2_dpesq_proxy.log and logs/r2_dpesq_proxy_files.csv.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT, load_config

from nrc.io import load_manifest, read_row, read_wav, resolve_data_root, split_root, to_pcm16
from nrc.metrics import pesq_wb

GATE = 0.6
LOG = REPO_ROOT / "logs" / "r2_dpesq_proxy.log"
CSV = REPO_ROOT / "logs" / "r2_dpesq_proxy_files.csv"
DSP_V8 = REPO_ROOT / "outputs" / "remix_pred_dsp_v8"


def out(msg: str = "") -> None:
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _pesq(args):
    clean, deg = args
    return pesq_wb(clean, deg, 16000)[0]


def q16(y):
    pcm, _ = to_pcm16(y)
    return pcm.astype(np.float64) / 32768.0


def corr(a, b) -> tuple[float, float, int]:
    from scipy.stats import pearsonr, spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(pearsonr(a[ok], b[ok])[0]), float(spearmanr(a[ok], b[ok])[0]), int(ok.sum())


def main() -> int:
    import torch
    from nrc.methods import get_method
    from nrc.tpesq import lfilter, pesq_loss

    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    torch.manual_seed(0)
    out(f"r2_dpesq_proxy {time.strftime('%Y-%m-%d %H:%M')} torch {torch.__version__}")
    loss_fn = pesq_loss(1.0, 16000)  # registers the torchaudio stub if needed, then imports torch_pesq
    import torch_pesq
    out(f"torch_pesq from {Path(torch_pesq.__file__).parent.name}; torchaudio stub: "
        f"{getattr(sys.modules.get('torchaudio'), '__nrc_stub__', False)}")

    # 1. shim check: FFT-convolution lfilter vs scipy.signal.lfilter (direct IIR) on a real signal
    from scipy.signal import lfilter as sp_lfilter
    root = resolve_data_root(None)
    vdf = load_manifest(root, "validation")
    x = read_row(root, vdf.iloc[0], "noisy")
    for name, coef in (("power_filter", loss_fn.power_filter), ("pre_filter", loss_fn.pre_filter)):
        b, a = coef[0].double().numpy(), coef[1].double().numpy()
        ref = sp_lfilter(b, a, x)
        got = lfilter(torch.from_numpy(x)[None], coef[1].double(), coef[0].double(), clamp=False)[0].numpy()
        got32 = lfilter(torch.from_numpy(x).float()[None], coef[1], coef[0], clamp=False)[0].double().numpy()
        out(f"lfilter shim {name}: max|err| float64 {np.max(np.abs(got - ref)):.2e}, float32 "
            f"{np.max(np.abs(got32 - ref)):.2e} (max|y| {np.max(np.abs(ref)):.2e})")

    # 2. per-file scores
    cfg = load_config(REPO_ROOT / "configs" / "dl" / "gru_v5b_off.yaml")
    enh = get_method(cfg["method"], **cfg["params"])
    rows, clean_l, deg_l = [], [], []
    t0 = time.perf_counter()
    for split in ("validation", "val_remix"):
        df = load_manifest(root, split)
        sroot = split_root(root, split)
        for _, r in df.iterrows():
            clean, noisy = read_row(sroot, r, "clean"), read_row(sroot, r, "noisy")
            degs = {"noisy": noisy, "v5b": q16(enh(noisy, 16000))}
            if split == "val_remix":
                degs["dspv8"] = read_wav(DSP_V8 / Path(r["noisy_path"]).name, 16000, r["num_samples"])
            for sys_name, d in degs.items():
                rows.append({"split": split, "segment_id": r["segment_id"], "system": sys_name,
                             "difficulty": r.get("noise_difficulty", "")})
                clean_l.append(clean)
                deg_l.append(d)
    out(f"enhanced + loaded {len(rows)} (file, system) pairs in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        p = list(pool.map(_pesq, zip(clean_l, deg_l), chunksize=8))
    out(f"pesq 0.0.4: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    mos = []
    with torch.no_grad():
        for i in range(0, len(clean_l), 32):
            c = torch.from_numpy(np.stack(clean_l[i:i + 32])).float()
            d = torch.from_numpy(np.stack(deg_l[i:i + 32])).float()
            mos.append(loss_fn.mos(c, d).double().numpy())
    out(f"torch-pesq MOS: {time.perf_counter() - t0:.1f}s")
    res = pd.DataFrame(rows)
    res["pesq"], res["tpesq_mos"] = p, np.concatenate(mos)
    res.to_csv(CSV, index=False, float_format="%.6f")

    out("\nmeans (pesq 0.0.4 / torch-pesq MOS):")
    for (split, s), g in res.groupby(["split", "system"]):
        out(f"  {split:10s} {s:6s} n {len(g):3d}  pesq {g.pesq.mean():.4f}  tpesq {g.tpesq_mos.mean():.4f}")

    out("\nper-file correlation, torch-pesq MOS vs pesq 0.0.4 (Pearson / Spearman / n):")
    for label, g in (("v5b", res[res.system == "v5b"]), ("noisy", res[res.system == "noisy"]),
                     ("dspv8 (val_remix)", res[res.system == "dspv8"]),
                     ("v5b + noisy pooled", res[res.system.isin(["v5b", "noisy"])])):
        pr, sr, n = corr(g.tpesq_mos, g.pesq)
        out(f"  {label:20s} {pr:.4f} / {sr:.4f} / {n}")

    w = res.pivot_table(index=["split", "segment_id"], columns="system", values=["pesq", "tpesq_mos"])
    out("\npaired-difference correlation, delta torch-pesq vs delta pesq (Pearson / Spearman / n):")
    gate = None
    for label, a, b, sel in (("v5b - noisy, all", "v5b", "noisy", None),
                             ("v5b - noisy, validation", "v5b", "noisy", "validation"),
                             ("v5b - noisy, val_remix", "v5b", "noisy", "val_remix"),
                             ("v5b - dspv8, val_remix", "v5b", "dspv8", "val_remix")):
        ww = w if sel is None else w.xs(sel, level="split", drop_level=False)
        dp = ww[("pesq", a)] - ww[("pesq", b)]
        dt = ww[("tpesq_mos", a)] - ww[("tpesq_mos", b)]
        pr, sr, n = corr(dt, dp)
        sign = float(np.mean(np.sign(dt[dp.notna() & dt.notna()]) == np.sign(dp[dp.notna() & dt.notna()])))
        out(f"  {label:26s} {pr:.4f} / {sr:.4f} / {n}  (sign agreement {sign:.3f}; mean dpesq {dp.mean():+.4f}, "
            f"dtpesq {dt.mean():+.4f})")
        if gate is None:
            gate = sr

    # 3. gradient checks
    out("\ngradient of PesqLoss forward(ref, deg).mean() w.r.t. deg (float32):")
    c0 = torch.from_numpy(np.stack(clean_l[:4])).float()
    d0 = torch.from_numpy(np.stack(deg_l[:4])).float()
    z = torch.zeros_like(c0)
    tiny = 1e-9 * torch.randn(c0.shape, generator=torch.Generator().manual_seed(0))
    cases = {"(a) deg silent, ref speech": (c0, z), "(b) ref silent, deg noisy speech": (z, d0),
             "(c) both silent": (z, z), "(d) normal 3 s (peak < 1)": (c0, d0),
             "(e) deg 1e-9 noise, ref speech": (c0, tiny), "(f) ref 1e-9 noise, deg speech": (tiny, d0)}
    for label, (ref, deg) in cases.items():
        deg = deg.clone().requires_grad_(True)
        loss = loss_fn(ref, deg).mean()
        loss.backward()
        gfin = bool(torch.isfinite(deg.grad).all())
        out(f"  {label:34s} loss {loss.item():.4f} finite loss {bool(torch.isfinite(loss))} finite grad {gfin} "
            f"|grad| {deg.grad.norm().item():.3e} peak in {float(max(ref.abs().max(), deg.detach().abs().max())):.3f}")
    out("  -> only exact digital silence is non-finite (0/0 in the level alignment); training drops rows whose clean or "
        "estimate has max|x| < 1e-8 before the PESQ term (train_dpesq.pesq_term)")

    verdict = "PASS" if gate is not None and gate >= GATE else "FAIL"
    out(f"\nGATE paired-difference Spearman (v5b - noisy, all) = {gate:.4f} vs {GATE}: {verdict}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
