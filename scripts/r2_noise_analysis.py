"""exp/r2-noise-analysis: where gru_v5b loses mean PESQ, by noise type and SNR (analysis only).

Pool = validation (164) + val_remix (328). Per file:
  * gru_v5b output exactly as main.py writes it (configs/dl/gru_v5b_off.yaml, 16-bit PCM): PESQ-WB, course SNR.
  * Shadow filtering (Loizou & Kim 2011): the same real mask M (v5b STFT 512/128 periodic Hann, centre/reflect)
    applied separately to the clean STFT S and the noise STFT N = STFT(noisy - clean):
      s_hat = iSTFT(M*S), n_hat = iSTFT(M*N).  s_hat + n_hat == v5b output up to float error (checked).
    PESQ(clean, s_hat) = distortion only; PESQ(clean, clean + n_hat) = residual noise only (both 16-bit).
    resid_db = 10 log10(sum n_hat^2 / sum n^2); speech_gain_db = 10 log10(sum s_hat^2 / sum s^2)
    (negative = speech attenuated).
  * Oracle IRM, same STFT and setup as the O-oracles of docs/exp_oracle.md (scipy Hann 1024 / hop 256,
    noisy phase, 100 Hz zero-phase 4th-order Butterworth high-pass, 16-bit): sqrt(|S|^2/(|S|^2+|N|^2)) (IRM,
    headline) and |S|^2/(|S|^2+|N|^2) (= O4, reference). headroom = PESQ(IRM) - PESQ(v5b).
  * Noise descriptors from n = noisy - clean (scipy STFT 512/128 Hann): stationarity, spectral centroid,
    low/high (< / >= 1 kHz) energy ratio, spectral flatness.

Stages:  compute -> outputs/r2_noise_analysis/files.csv (git-ignored, heavy, one process pool)
         report  -> results/r2_noise_analysis_files.csv, results/r2_noise_analysis_types.csv,
                    logs/r2_noise_analysis_tables.md, docs/figures/r2_noise_*.png
Usage: python scripts/r2_noise_analysis.py --stage compute --jobs 4
       python scripts/r2_noise_analysis.py --stage report
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT, git_hash, load_config

from nrc.io import load_manifest, read_row, resolve_data_root, split_root, to_pcm16
from nrc.methods import get_method
from nrc.methods import mmse_lsa as ml
from nrc.metrics import pesq_wb, snr_db
from nrc.stft import istft, stft

CFG = REPO_ROOT / "configs" / "dl" / "gru_v5b_off.yaml"
OUT = REPO_ROOT / "outputs" / "r2_noise_analysis"
META = ("segment_id", "noise_category", "noise_collection", "noise_code", "noise_difficulty", "target_snr_db")
PESQ_MAX = 4.64
ORACLE = {"n_fft": 1024, "hop": 256, "window": "hann", "highpass_hz": 100}
EPS = 1e-12
N_BANDS = 16  # log-spaced 100 Hz .. 8 kHz bands for the stationarity descriptor
_STATE = {}


def q16(y):
    return to_pcm16(y)[0].astype(np.float64) / 32768.0


def _model():
    if "model" not in _STATE:
        import torch
        torch.set_num_threads(1)
        from nrc.models.gru_mask import load_weights
        cfg = load_config(CFG)
        _STATE["method"] = get_method(cfg["method"], **cfg["params"])
        _STATE["model"] = load_weights(REPO_ROOT / cfg["params"]["weights"], "cpu")
    return _STATE["model"], _STATE["method"]


def noise_descriptors(n):
    f, _, z = stft(n, 16000, 512, 128, "hann")
    p = np.abs(z) ** 2 + EPS                                    # (F, T)
    edges = np.geomspace(100, 8000, N_BANDS + 1)
    band = np.stack([p[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(edges[:-1], edges[1:])])
    active = p.sum(0) > 1e-10 * p.sum(0).max()                 # drop exactly silent (padding) frames
    stationarity = float(np.mean(np.std(10 * np.log10(band[:, active]), axis=1)))
    mean_psd = p[:, active].mean(1)
    centroid = float(np.sum(f * mean_psd) / np.sum(mean_psd))
    lo_hi_db = float(10 * np.log10(mean_psd[f < 1000].sum() / mean_psd[f >= 1000].sum()))
    sel = f >= 50
    flat = float(np.exp(np.mean(np.log(mean_psd[sel]))) / np.mean(mean_psd[sel]))
    flat_frames = np.exp(np.mean(np.log(p[sel][:, active]), 0)) / np.mean(p[sel][:, active], 0)
    return {"nd_stationarity_db": stationarity, "nd_centroid_hz": centroid, "nd_lo_hi_db": lo_hi_db,
            "nd_flatness": flat, "nd_flatness_frames": float(np.mean(flat_frames))}


def run_file(task):
    import torch
    root, row, split = task
    model, method = _model()
    noisy, clean = read_row(root, row, "noisy"), read_row(root, row, "clean")
    noise = noisy - clean
    n = len(noisy)
    assert n > model.n_fft
    # --- gru_v5b: registered method (what main.py writes) and the explicit mask path ---------
    y_reg = method(noisy, 16000)
    with torch.inference_mode():
        def t(x):
            return torch.from_numpy(x.astype(np.float32))[None]
        spec_x = model.stft(t(noisy))
        m = model.mask(spec_x.abs())
        y_mask = model.istft(m * spec_x, length=n)[0].double().numpy()
        s_hat = model.istft(m * model.stft(t(clean)), length=n)[0].double().numpy()
        n_hat = model.istft(m * model.stft(t(noise)), length=n)[0].double().numpy()
    y16 = q16(y_reg)
    rec = {c: row[c] for c in META}
    rec["source"] = split
    rec["maxdiff_mask_vs_method"] = float(np.max(np.abs(y_mask - y_reg)))
    rec["maxdiff_shadow_sum"] = float(np.max(np.abs(s_hat + n_hat - y_reg)))
    rec["maxdiff_shadow_sum_16bit"] = float(np.max(np.abs(q16(s_hat + n_hat) - y16)))
    rec["mask_mean"] = float(m.mean())
    rec["pesq_noisy"] = pesq_wb(clean, noisy, 16000)[0]
    rec["pesq"], rec["pesq_error"] = pesq_wb(clean, y16, 16000)
    rec["snr_db"] = snr_db(clean, y16)
    rec["pesq_dist"] = pesq_wb(clean, q16(s_hat), 16000)[0]
    rec["pesq_resid"] = pesq_wb(clean, q16(clean + n_hat), 16000)[0]
    rec["resid_db"] = float(10 * np.log10(np.sum(n_hat ** 2) / np.sum(noise ** 2)))
    rec["speech_gain_db"] = float(10 * np.log10(np.sum(s_hat ** 2) / np.sum(clean ** 2)))
    rec["snr_dist_db"] = snr_db(clean, s_hat)
    # --- oracle masks, O-oracle setup ---------------------------------------------------------
    o = ORACLE
    _, _, z = stft(noisy, 16000, o["n_fft"], o["hop"], o["window"])
    _, _, s = stft(clean, 16000, o["n_fft"], o["hop"], o["window"])
    _, _, nn = stft(noise, 16000, o["n_fft"], o["hop"], o["window"])
    s2, n2 = np.abs(s) ** 2, np.abs(nn) ** 2
    wien = s2 / np.maximum(s2 + n2, ml._EPS)
    for k, g in (("irm", np.sqrt(wien)), ("wiener", wien)):
        y = ml._highpass(istft(g * z, n, 16000, o["n_fft"], o["hop"], o["window"]), o["highpass_hz"], 16000)
        y = q16(y)
        rec[f"pesq_{k}"] = pesq_wb(clean, y, 16000)[0]
        rec[f"snr_{k}"] = snr_db(clean, y)
    rec.update(noise_descriptors(noise))
    return rec


def compute(args):
    root = resolve_data_root(args.data_root)
    tasks = []
    for split in ("validation", "val_remix"):
        man = load_manifest(root, split)
        sroot = split_root(root, split)
        tasks += [(sroot, r.to_dict(), split) for _, r in man.iterrows()]
    if args.limit:
        tasks = tasks[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for i, r in enumerate(ex.map(run_file, tasks, chunksize=4), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "code_hash", git_hash())
    df.to_csv(OUT / ("files.csv" if not args.limit else "files_limit.csv"), index=False)
    print(f"wrote {len(df)} rows, {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------
def ci95(x):
    from scipy import stats
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def md(d, fmt):
    """DataFrame -> GitHub markdown table (no tabulate dependency)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else format(v, fmt)
        return str(v)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def group_table(df, by, n_total, h_total):
    rows = []
    for key, g in df.groupby(by):
        dist_drop = (PESQ_MAX - g.pesq_dist).mean()
        res_drop = (PESQ_MAX - g.pesq_resid).mean()
        rows.append({
            **(dict(zip(by, key)) if isinstance(by, list) else {by: key}),
            "n": len(g), "pesq": g.pesq.mean(), "ci95": ci95(g.pesq),
            "pesq_noisy": g.pesq_noisy.mean(), "pesq_irm": g.pesq_irm.mean(),
            "deficit_contrib": (PESQ_MAX - g.pesq).sum() / n_total,
            "headroom_mean": g.headroom.mean(), "headroom_share": g.headroom.sum() / h_total,
            "headroom_wiener_share": g.headroom_wiener.sum() / df.attrs["hw_total"],
            "pesq_dist": g.pesq_dist.mean(), "pesq_resid": g.pesq_resid.mean(),
            "dist_drop": dist_drop, "resid_drop": res_drop,
            "dominant": "distortion" if dist_drop > res_drop else "residual",
            "resid_db": g.resid_db.mean(), "speech_gain_db": g.speech_gain_db.mean(),
            "snr_db": g.snr_db.mean(),
        })
    return pd.DataFrame(rows)


def report(args):
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT / "files.csv")
    out = []

    def p(s=""):
        out.append(s)

    hashes = df.code_hash.unique().tolist()
    df["headroom"] = df.pesq_irm - df.pesq
    df["headroom_wiener"] = df.pesq_wiener - df.pesq
    n_total, h_total = len(df), df.headroom.sum()
    df.attrs["hw_total"] = df.headroom_wiener.sum()
    p(f"code hash {hashes}; n = {n_total}; failed PESQ: {int(df.pesq.isna().sum())}")

    # reproduction of the recorded eval.py per-file results
    for split in ("validation", "val_remix"):
        ref = pd.read_csv(REPO_ROOT / "outputs" / "gru_v5b_off" / f"eval_{split}.csv").set_index("segment_id")
        d = df[df.source == split].set_index("segment_id")
        p(f"{split}: mean PESQ {d.pesq.mean():.4f} (eval.py {ref.pesq.mean():.4f}), "
          f"max |dPESQ| {np.max(np.abs(d.pesq - ref.pesq.reindex(d.index))):.2e}, "
          f"max |dSNR| {np.max(np.abs(d.snr_db - ref.snr_db.reindex(d.index))):.2e}")
    p(f"consistency: max |mask path - registered method| {df.maxdiff_mask_vs_method.max():.3e}; "
      f"max |s_hat + n_hat - v5b| {df.maxdiff_shadow_sum.max():.3e} (float), "
      f"{df.maxdiff_shadow_sum_16bit.max():.3e} after 16-bit (1 LSB = {1/32768:.3e})")

    p("\n## Overall (pool and per source)")
    hdr = "| set | n | v5b PESQ ± CI | noisy | IRM oracle | Wiener (O4) | headroom IRM | dist-only PESQ | resid-only PESQ | dist drop | resid drop | resid dB | speech dB | SNR dB |"
    p(hdr)
    p("|" + "|".join(["---"] * 14) + "|")
    for name, g in [("pool", df)] + list(df.groupby("source")):
        p(f"| {name} | {len(g)} | {g.pesq.mean():.4f} ± {ci95(g.pesq):.4f} | {g.pesq_noisy.mean():.3f} | "
          f"{g.pesq_irm.mean():.3f} | {g.pesq_wiener.mean():.3f} | {g.headroom.mean():+.3f} | {g.pesq_dist.mean():.3f} | "
          f"{g.pesq_resid.mean():.3f} | {(PESQ_MAX - g.pesq_dist).mean():.3f} | {(PESQ_MAX - g.pesq_resid).mean():.3f} | "
          f"{g.resid_db.mean():.2f} | {g.speech_gain_db.mean():.2f} | {g.snr_db.mean():.2f} |")
    frac_dist = float(np.mean(df.pesq_dist < df.pesq_resid))
    p(f"\nfiles where distortion-only PESQ < residual-only PESQ (distortion dominant): {frac_dist:.3f}; "
      f"files with headroom < 0: {int((df.headroom < 0).sum())}")

    snr = group_table(df, "target_snr_db", n_total, h_total)
    p("\n## Per SNR")
    p(md(snr, ".3f"))
    diff = group_table(df, "noise_difficulty", n_total, h_total)
    p("\n## Per difficulty")
    p(md(diff, ".3f"))

    typ = group_table(df, "noise_category", n_total, h_total).sort_values("headroom_share", ascending=False)
    coll = df.groupby("noise_category").noise_collection.first()
    typ.insert(1, "collection", typ.noise_category.map(coll))
    nd_cols = [c for c in df.columns if c.startswith("nd_")]
    nd = df.groupby("noise_category")[nd_cols].mean()
    typ = typ.merge(nd, left_on="noise_category", right_index=True)
    p("\n## Per noise type, ranked by recoverable-headroom share")
    p(md(typ.drop(columns=nd_cols), ".3f"))

    ts = group_table(df, ["noise_category", "target_snr_db"], n_total, h_total)
    p("\n## Type x SNR headroom share (%)")
    piv = ts.pivot(index="noise_category", columns="target_snr_db", values="headroom_share") * 100
    piv = piv.reindex(typ.noise_category)
    p(md(piv.reset_index(), ".2f"))
    p("\n## Type x SNR mean headroom (IRM - v5b)")
    pivm = ts.pivot(index="noise_category", columns="target_snr_db", values="headroom_mean").reindex(typ.noise_category)
    p(md(pivm.reset_index(), ".3f"))
    p("\n## Type x SNR n")
    p(md(ts.pivot(index="noise_category", columns="target_snr_db", values="n").reindex(typ.noise_category)
      .reset_index(), ".0f"))

    p("\n## Noise descriptors per type and Spearman correlation with per-type mean headroom (23 types)")
    p(md(typ[["noise_category", "n", "headroom_mean"] + nd_cols], ".3f"))
    p("")
    p("| descriptor | rho vs type mean headroom | p | rho vs type headroom share | p | rho per file (n=492) | p |")
    p("|---|---|---|---|---|---|---|")
    corr = {}
    for c in nd_cols:
        r1 = stats.spearmanr(typ[c], typ.headroom_mean)
        r2 = stats.spearmanr(typ[c], typ.headroom_share)
        r3 = stats.spearmanr(df[c], df.headroom)
        corr[c] = r1.statistic
        p(f"| {c} | {r1.statistic:+.3f} | {r1.pvalue:.3f} | {r2.statistic:+.3f} | {r2.pvalue:.3f} | "
          f"{r3.statistic:+.3f} | {r3.pvalue:.2g} |")
    # within-SNR control: per-file partial (rank) correlation at each SNR
    p("\nper-file Spearman within each SNR (descriptor vs headroom):")
    for s, g in df.groupby("target_snr_db"):
        p(f"  {s:g} dB (n={len(g)}): " + ", ".join(
            f"{c[3:]} {stats.spearmanr(g[c], g.headroom).statistic:+.2f}" for c in nd_cols))

    # outputs
    (REPO_ROOT / "results").mkdir(exist_ok=True)
    keep = ["code_hash", "source", *META, "pesq_noisy", "pesq", "snr_db", "pesq_dist", "pesq_resid", "resid_db",
            "speech_gain_db", "snr_dist_db", "pesq_irm", "snr_irm", "pesq_wiener", "snr_wiener", "headroom", "headroom_wiener",
            "maxdiff_shadow_sum", "maxdiff_shadow_sum_16bit", "mask_mean", *nd_cols]
    df[keep].to_csv(REPO_ROOT / "results" / "r2_noise_analysis_files.csv", index=False, float_format="%.6g")
    typ.to_csv(REPO_ROOT / "results" / "r2_noise_analysis_types.csv", index=False, float_format="%.6g")
    (REPO_ROOT / "logs").mkdir(exist_ok=True)
    (REPO_ROOT / "logs" / "r2_noise_analysis_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))

    figs = REPO_ROOT / "docs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "text.color": "#0b0b0b",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.edgecolor": "#b5b4ad"})
    blues = LinearSegmentedColormap.from_list(
        "ramp_blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    # 1) heat map: mean headroom per type x SNR, n annotated
    nmat = ts.pivot(index="noise_category", columns="target_snr_db", values="n").reindex(typ.noise_category)
    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    im = ax.imshow(pivm.values, cmap=blues, aspect="auto", vmin=0)
    ax.set_xticks(range(pivm.shape[1]), [f"{c:g} dB" for c in pivm.columns])
    ylab = [f"{c} ({s*100:.1f}%)" for c, s in zip(typ.noise_category, typ.headroom_share)]
    ax.set_yticks(range(pivm.shape[0]), ylab)
    vmax = np.nanmax(pivm.values)
    for i in range(pivm.shape[0]):
        for j in range(pivm.shape[1]):
            v, k = pivm.values[i, j], nmat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}\nn={k:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 * vmax else "#0b0b0b")
    ax.spines[:].set_visible(False)
    ax.set_title("gru_v5b recoverable headroom: PESQ(oracle IRM) - PESQ(v5b)\n"
                 "validation + val_remix (n=492); rows ranked by total headroom share (in label)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="mean PESQ headroom")
    fig.tight_layout()
    fig.savefig(figs / "r2_noise_headroom_heatmap.png", dpi=150)
    plt.close(fig)
    # 2) per-type bars: distortion-only vs residual-only PESQ drop
    t2 = typ.sort_values("headroom_share", ascending=True)
    y = np.arange(len(t2))
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.barh(y + 0.2, t2.dist_drop, height=0.4, color="#2a78d6", label="distortion only: 4.64 - PESQ(clean, s_hat)")
    ax.barh(y - 0.2, t2.resid_drop, height=0.4, color="#eb6834", label="residual noise only: 4.64 - PESQ(clean, clean + n_hat)")
    ax.set_yticks(y, [f"{c} (n={k})" for c, k in zip(t2.noise_category, t2.n)])
    ax.set_xlabel("mean PESQ drop from 4.64 (higher = worse)")
    ax.set_title("gru_v5b shadow filtering per noise type (validation + val_remix)\n"
                 "rows ranked by recoverable-headroom share, largest at top", fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figs / "r2_noise_dist_vs_resid.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("compute", "report"), required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N files only")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)
    compute(args) if args.stage == "compute" else report(args)


if __name__ == "__main__":
    sys.exit(main())
