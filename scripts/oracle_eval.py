"""Oracle (clean-informed) diagnostics for the dsp-v6 MMSE-LSA/HRNR pipeline. ANALYSIS ONLY.

This script reads the clean reference and the true noise (noisy - clean) to build oracle variants.
It is NOT a method: it must never be registered in src/nrc/methods nor imported by main.py.
Results go to logs/oracle_*.{csv,txt}; nothing is appended to results/experiments.csv.

Variants (dsp-v6 = configs/dsp_h.yaml settings everywhere except the oracle part):
  O0      dsp-v6 as is (via the registered method; must reproduce scripts/eval.py)
  O1_b1   oracle stationary noise PSD: per-bin mean of |N(k,l)|^2 over all frames, bias 1.0
  O1_b2   same, with the dsp-v6 two-band bias (noise_bias_low / noise_bias_high)
  O2_b1   oracle time-varying noise PSD: |N(k,l)|^2 averaged over +-2 frames, bias 1.0
  O2_b2   same, with the dsp-v6 two-band bias
  O3      dsp-v6 gain * |Y| combined with the clean STFT phase
  O4      ideal Wiener mask |S|^2/(|S|^2+|N|^2) on the noisy STFT (noisy phase), then the high-pass
  O5_b1   O2_b1 gain with the clean STFT phase
  O5_b2   O2_b2 gain with the clean STFT phase
All outputs are quantized to 16-bit PCM before scoring (pesq 0.0.4 'wb', course SNR).

--lsd-grid <sweep yaml>: instead of the variants, per file, compare the stationary noise PSD (before any
noise_bias / two-band bias) of --config and of every grid point of the sweep yaml with the oracle mean noise
PSD mean_l |N(k,l)|^2: LSD = RMS over bins 100 Hz-8 kHz of 10 log10(est/oracle) (dB), signed bias = mean of
the same (dB; < 0 = under-estimation). Output logs/oracle_lsd_<tag>.{csv,txt}. No PESQ.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from _common import REPO_ROOT, git_hash, load_config

from nrc.io import load_manifest, read_row, resolve_data_root, split_root, stratified_subset, to_pcm16
from nrc.methods import get_method
from nrc.methods import mmse_lsa as ml
from nrc.metrics import pesq_wb, snr_db
from nrc.stft import istft, stft

META = ("segment_id", "noise_category", "noise_code", "noise_difficulty", "target_snr_db")
VARIANTS = ("O0", "O1_b1", "O1_b2", "O2_b1", "O2_b2", "O3", "O4", "O5_b1", "O5_b2")
_CACHE = {}


def v6_gain(z, noise, p):
    """dsp-v6 final gain (DD -> TSNR -> HRNR, LSA) for a given noise PSD (bins, frames).

    Mirrors mmse_lsa._enhance_two_step for xi_mode 'hrnr', no cepstral smoothing, no pause.
    """
    n_fft, hop, win, sr = p["n_fft"], p["hop"], p["window"], 16000
    noise = np.maximum(noise, ml._EPS)
    xi_min, gmin = 10.0 ** (p["xi_min_db"] / 10.0), 10.0 ** (p["gmin_db"] / 20.0)
    alpha, rho = p["alpha"], p["hrnr_rho"]
    gamma = np.minimum(np.abs(z) ** 2 / noise, 1e8)
    g_final = np.empty(z.shape)
    prev = np.zeros(z.shape[0])
    for t in range(z.shape[1]):
        gm = gamma[:, t]
        inst = np.maximum(gm - 1.0, 0.0)
        xi1 = alpha + (1 - alpha) * inst if t == 0 else alpha * prev / noise[:, t] + (1 - alpha) * inst
        g = np.clip(ml.lsa_gain(np.maximum(xi1, xi_min), gm), gmin, 1.0)
        g = np.clip(ml.lsa_gain(np.maximum(g * g * gm, xi_min), gm), gmin, 1.0)
        g_final[:, t] = g
        prev = g * g * gm * noise[:, t]
    length = p["_length"]
    s_tsnr = istft(g_final * z, length, sr, n_fft, hop, win)
    _, _, z_h = stft(np.maximum(s_tsnr, 0.0), sr, n_fft, hop, win)
    z_h = z_h[:, :z.shape[1]]
    xi_h = (rho * g_final ** 2 * np.abs(z) ** 2 + (1 - rho) * np.abs(z_h) ** 2) / noise
    return np.clip(ml.lsa_gain(np.maximum(xi_h, xi_min), gamma), gmin, 1.0)


def synth(spec, p):
    y = istft(spec, p["_length"], 16000, p["n_fft"], p["hop"], p["window"])
    return ml._highpass(y, p["highpass_hz"], 16000)


def run_file(task):
    root, row, cfg = task
    p = dict(cfg["params"])
    noisy, clean = read_row(root, row, "noisy"), read_row(root, row, "clean")
    p["_length"] = len(noisy)
    n_fft, hop, win = p["n_fft"], p["hop"], p["window"]
    key = cfg["name"]
    if key not in _CACHE:
        _CACHE[key] = get_method(cfg["method"], **cfg["params"])
    _, times, z = stft(noisy, 16000, n_fft, hop, win)
    _, _, s = stft(clean, 16000, n_fft, hop, win)
    _, _, n = stft(noisy - clean, 16000, n_fft, hop, win)
    two_band = ml.two_band_bias(z.shape[0], 16000, n_fft, p["noise_bias_low"],
                                p["noise_bias_high"])[:, None]
    n2, s2 = np.abs(n) ** 2, np.abs(s) ** 2
    clean_phase = np.exp(1j * np.angle(s))

    outs = {"O0": _CACHE[key](noisy, 16000)}
    # O0 re-implementation check (estimated PSD through v6_gain) == registered method
    est = ml.banded_initial_noise(z, times, len(noisy), 16000, n_fft, p["noise_fraction"], 1)
    g0 = v6_gain(z, np.repeat(est[:, None], z.shape[1], 1) * two_band, p)
    o0_diff = float(np.max(np.abs(synth(g0 * z, p) - outs["O0"])))
    stat = np.repeat(n2.mean(axis=1, keepdims=True), z.shape[1], 1)
    tv = uniform_filter1d(n2, size=5, axis=1, mode="nearest")
    gains = {"O1_b1": v6_gain(z, stat, p), "O1_b2": v6_gain(z, stat * two_band, p),
             "O2_b1": v6_gain(z, tv, p), "O2_b2": v6_gain(z, tv * two_band, p)}
    for k, g in gains.items():
        outs[k] = synth(g * z, p)
    outs["O3"] = synth(g0 * np.abs(z) * clean_phase, p)
    outs["O4"] = synth(s2 / np.maximum(s2 + n2, ml._EPS) * z, p)
    outs["O5_b1"] = synth(gains["O2_b1"] * np.abs(z) * clean_phase, p)
    outs["O5_b2"] = synth(gains["O2_b2"] * np.abs(z) * clean_phase, p)

    rec = {c: row[c] for c in META}
    rec["o0_reimpl_maxdiff"] = o0_diff
    for k in VARIANTS:
        pcm, _ = to_pcm16(outs[k])
        y = pcm.astype(np.float64) / 32768.0
        rec[f"pesq_{k}"] = pesq_wb(clean, y, 16000)[0]
        rec[f"snr_{k}"] = snr_db(clean, y)
    return rec


_LSD_KEYS = ("noise_est", "noise_fraction", "q", "bias", "smooth_frames", "noise_select_bands",
             "noise_freq_smooth_bins", "noise_freq_smooth_mode", "spp_iters", "spp_xi_h1_db",
             "spp_prior")


def lsd_file(task):
    """Per-file LSD / signed bias (dB) of each estimator's stationary pre-bias PSD vs the oracle mean."""
    root, row, base, variants = task
    noisy, clean = read_row(root, row, "noisy"), read_row(root, row, "clean")
    n_fft, hop, win = base["n_fft"], base["hop"], base["window"]
    _, times, z = stft(noisy, 16000, n_fft, hop, win)
    _, _, n = stft(noisy - clean, 16000, n_fft, hop, win)
    power = np.abs(z) ** 2
    oracle = np.maximum((np.abs(n) ** 2).mean(axis=1), ml._EPS)
    freqs = np.arange(z.shape[0]) * 16000 / n_fft
    band = (freqs >= 100.0) & (freqs <= 8000.0)
    rec = {c: row[c] for c in META}
    for name, p in variants:
        kw = {k: p[k] for k in _LSD_KEYS if k in p}
        est = np.maximum(ml.stationary_noise_psd(z, power, times, len(noisy), 16000, n_fft, **kw), ml._EPS)
        e = 10.0 * np.log10(est[band] / oracle[band])
        rec[f"lsd_{name}"] = float(np.sqrt(np.mean(e ** 2)))
        rec[f"bias_{name}"] = float(np.mean(e))
    return rec


def run_lsd(a, cfg, root, man):
    import yaml
    from sweep import grid_points
    spec = yaml.safe_load((REPO_ROOT / a.lsd_grid).read_text(encoding="utf-8"))
    variants = [(cfg["name"], dict(cfg["params"]))]
    for pt in grid_points(spec):
        diff = {k: v for k, v in pt.items() if k not in spec.get("base_params", {})}
        variants.append(("|".join(f"{k}={v}" for k, v in diff.items()), pt))
    tasks = [(root, r.to_dict(), cfg["params"], variants) for _, r in man.iterrows()]
    t0 = time.perf_counter()
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            rows = list(pool.map(lsd_file, tasks, chunksize=4))
    else:
        rows = [lsd_file(t) for t in tasks]
    d = pd.DataFrame(rows)
    names = [v[0] for v in variants]
    t = pd.DataFrame({"LSD dB": [d[f"lsd_{k}"].mean() for k in names],
                      "signed bias dB": [d[f"bias_{k}"].mean() for k in names]}, index=names)
    lines = [t.to_string(float_format=lambda v: f"{v:+.3f}")]
    for by in ("noise_difficulty",):
        g = d.groupby(by)
        tt = pd.DataFrame({k: g[f"lsd_{k}"].mean() for k in names}).T
        lines.append(f"\n[LSD dB by {by}]\n" + tt.to_string(float_format=lambda v: f"{v:.3f}"))
        tt = pd.DataFrame({k: g[f"bias_{k}"].mean() for k in names}).T
        lines.append(f"\n[signed bias dB by {by}]\n" + tt.to_string(float_format=lambda v: f"{v:+.3f}"))
    tag = f"{a.split}" + (f"_sub{a.subset}" if a.subset else "")
    head = (f"oracle_eval --lsd-grid {a.lsd_grid} base {cfg['name']} | {tag} n={len(d)} | hash {git_hash()} | "
            f"{time.perf_counter() - t0:.0f}s | band 100 Hz-8 kHz, pre-bias PSD vs oracle mean |N|^2")
    text = head + "\n" + "\n".join(lines)
    logs = REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    stem = f"oracle_lsd_{Path(a.lsd_grid).stem}_{tag}"
    d.to_csv(logs / f"{stem}.csv", index=False)
    (logs / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def tables(d):
    lines = []
    for by in (None, "target_snr_db", "noise_difficulty"):
        g = d.groupby(by) if by else d.groupby(lambda _: "all")
        t = pd.DataFrame({"n": g.size()})
        for k in VARIANTS:
            t[k] = g[f"pesq_{k}"].mean()
        lines.append(f"\n[PESQ by {by or 'overall'}]\n" + t.to_string(float_format=lambda v: f"{v:.4f}"))
    snr = pd.Series({k: d[f"snr_{k}"].mean() for k in VARIANTS}, name="SNR dB")
    lines.append("\n[SNR dB overall]\n" + snr.to_string(float_format=lambda v: f"{v:.2f}"))
    rows = []
    for k in VARIANTS[1:]:
        diff = d[f"pesq_{k}"] - d["pesq_O0"]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append((k, diff.mean(), se, diff.mean() / se if se > 0 else np.nan))
    pairs = [("(a) O1_b1-O0", "O1_b1", "O0"), ("(a') O1_b2-O0", "O1_b2", "O0"),
             ("(b) O2_b1-O1_b1", "O2_b1", "O1_b1"), ("(b') O2_b2-O1_b2", "O2_b2", "O1_b2"),
             ("(c) O3-O0", "O3", "O0"), ("O5_b1-O2_b1", "O5_b1", "O2_b1")]
    for name, a, b in pairs:
        diff = d[f"pesq_{a}"] - d[f"pesq_{b}"]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append((name, diff.mean(), se, diff.mean() / se))
    lines.append("\n[paired PESQ deltas]\n" + pd.DataFrame(rows, columns=["delta", "mean", "SE", "z"])
                 .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    for k in ("O1_b1", "O1_b2", "O2_b1", "O2_b2"):
        g = (d[f"pesq_{k}"] - d["pesq_O0"]).groupby(d.noise_category)
        t = pd.DataFrame({"n": g.size(), "gain": g.mean()}).sort_values("gain", ascending=False).head(5)
        lines.append(f"\n[top-5 noise categories, {k} - O0]\n"
                     + t.to_string(float_format=lambda v: f"{v:+.4f}"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dsp_h.yaml")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--lsd-grid", default=None,
                    help="sweep yaml: PSD log-spectral distance diagnostic instead of the O-variants")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    assert cfg["method"] == "mmse_lsa" and cfg["params"]["xi_mode"] == "hrnr"
    root = resolve_data_root(a.data_root)
    man = stratified_subset(load_manifest(root, a.split), a.subset)
    root = split_root(root, a.split)
    if a.lsd_grid:
        return run_lsd(a, cfg, root, man)
    tasks = [(root, r.to_dict(), cfg) for _, r in man.iterrows()]
    t0 = time.perf_counter()
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            rows = list(pool.map(run_file, tasks, chunksize=4))
    else:
        rows = [run_file(t) for t in tasks]
    d = pd.DataFrame(rows)
    tag = f"{a.split}" + (f"_sub{a.subset}" if a.subset else "")
    head = (f"oracle_eval {cfg['name']} | {tag} n={len(d)} | hash {git_hash()} | "
            f"{time.perf_counter() - t0:.0f}s | PESQ NaN={int(d.filter(like='pesq_').isna().sum().sum())} | "
            f"O0 re-impl max |diff| = {d.o0_reimpl_maxdiff.max():.2e}")
    text = head + "\n" + tables(d)
    logs = REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    d.to_csv(logs / f"oracle_{tag}.csv", index=False)
    (logs / f"oracle_{tag}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
