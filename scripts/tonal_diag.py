"""Tonal-track noise-marking diagnostic for noise_tonal (dsp-o). ANALYSIS ONLY.

Reads the clean reference (and noise = noisy - clean) to label tracks; it is NOT a method and must never be
imported by src/nrc or main.py. Nothing is appended to results/experiments.csv.

Files: the stratified --subset of train (default 20, seed 0) plus every train file whose noise_category is
in --cats (default alarm_siren, phone_dialing). Per file, with --config's STFT and stationary estimate
(spp_mean), the peak tracks of mmse_lsa.tonal_tracks at each prominence in --prom are listed; candidate tracks
are those lasting >= tonal_min_frames. Per track: duration, speech-absent frames in the span, detected
speech-absent frames, speech energy fraction sum|S|^2 / (sum|S|^2 + sum|N|^2) over the track cells
(bins kb +- tonal_bins), and the noise marking under the brief rule (min absent frames 0, span 0) and the
tightened rule (8, 0.3). Ground truth: a track is noise if its speech fraction < 0.5.
Precision = truly-noise share of marked tracks; recall = marked share of truly-noise candidate tracks.
Output logs/tonal_diag.{csv,txt}.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from _common import REPO_ROOT, git_hash, load_config

from nrc.io import load_manifest, read_row, resolve_data_root, split_root, stratified_subset
from nrc.methods import mmse_lsa as ml
from nrc.stft import stft

RULES = {"brief": (0, 0.0), "tight": (8, 0.3)}


def diag_file(task):
    root, row, p, proms, frac, bins, min_frames, link = task
    noisy, clean = read_row(root, row, "noisy"), read_row(root, row, "clean")
    n_fft, hop, win = p["n_fft"], p["hop"], p["window"]
    _, times, z = stft(noisy, 16000, n_fft, hop, win)
    _, _, s = stft(clean, 16000, n_fft, hop, win)
    _, _, n = stft(noisy - clean, 16000, n_fft, hop, win)
    power, s2, n2 = np.abs(z) ** 2, np.abs(s) ** 2, np.abs(n) ** 2
    stat = ml.stationary_noise_psd(z, power, times, len(noisy), 16000, n_fft, p["noise_est"],
                                   p["noise_fraction"], spp_iters=p["spp_iters"],
                                   spp_xi_h1_db=p["spp_xi_h1_db"])
    absent = ml.speech_absent_frames(power, stat, 16000, n_fft, p["spp_xi_h1_db"])
    recs = []
    for prom in proms:
        for frames, kb, det in ml.tonal_tracks(power, 16000, n_fft, prom, link):
            if len(frames) < min_frames:
                continue
            k = np.clip(kb[None, :] + np.arange(-bins, bins + 1)[:, None], 0, power.shape[0] - 1)
            f = np.broadcast_to(frames, k.shape)
            es, en = s2[k, f].sum(), n2[k, f].sum()
            ab = absent[frames]
            rec = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
                   "prom_db": prom, "dur": len(frames), "n_absent": int(ab.sum()),
                   "det_absent": int((ab & det).sum()), "speech_frac": es / max(es + en, ml._EPS),
                   "mean_hz": float(kb.mean() * 16000 / n_fft)}
            for name, (mf, span) in RULES.items():
                rec[f"noise_{name}"] = ml.tonal_track_is_noise(frames, det, absent, frac, min_frames,
                                                               mf, span)
            recs.append(rec)
    return recs


def summary(d, group):
    rows = []
    for (g, prom), x in d.groupby([group, "prom_db"]):
        truth = x.speech_frac < 0.5
        r = {group: g, "prom": prom, "files": x.segment_id.nunique(), "tracks": len(x),
             "true_noise": int(truth.sum()), "med_dur": x.dur.median(),
             "med_absent": x.n_absent.median()}
        for name in RULES:
            m = x[f"noise_{name}"]
            tp = int((m & truth).sum())
            r[f"{name}_marked"] = int(m.sum())
            r[f"{name}_prec"] = tp / m.sum() if m.sum() else np.nan
            r[f"{name}_rec"] = tp / truth.sum() if truth.sum() else np.nan
            r[f"{name}_spfrac"] = x.loc[m, "speech_frac"].mean() if m.sum() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dsp_k.yaml")
    ap.add_argument("--subset", type=int, default=20)
    ap.add_argument("--cats", nargs="*", default=["alarm_siren", "phone_dialing"])
    ap.add_argument("--prom", nargs="*", type=float, default=[6.0, 10.0, 14.0])
    ap.add_argument("--absent-frac", type=float, default=0.7)
    ap.add_argument("--bins", type=int, default=1)
    ap.add_argument("--min-frames", type=int, default=12)
    ap.add_argument("--link-bins", type=int, default=2)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    p = load_config(a.config)["params"]
    root = resolve_data_root(a.data_root)
    man = load_manifest(root, "train")
    sub = stratified_subset(man, a.subset).assign(group=f"sub{a.subset}")
    cat = man[man.noise_category.isin(a.cats)].assign(group=man.noise_category)
    files = pd.concat([sub, cat])
    root = split_root(root, "train")
    tasks = [(root, r.to_dict(), p, a.prom, a.absent_frac, a.bins, a.min_frames, a.link_bins)
             for _, r in files.drop_duplicates("segment_id").iterrows()]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        d = pd.DataFrame([r for rs in pool.map(diag_file, tasks, chunksize=2) for r in rs])
    groups = files[["segment_id", "group"]]
    d = d.merge(groups, on="segment_id")  # a subset file in a cat appears in both groups
    fmt = {"index": False, "float_format": lambda v: f"{v:.3f}"}
    head = (f"tonal_diag {a.config} | train files {len(tasks)} | hash {git_hash()} | "
            f"{time.perf_counter() - t0:.0f}s | absent_frac {a.absent_frac} bins {a.bins} min_frames "
            f"{a.min_frames} | candidate tracks = dur >= min_frames; truth noise = speech_frac < 0.5 | "
            f"rules brief (0, 0) / tight (min absent frames 8, absent span 0.3)")
    text = head + "\n" + summary(d, "group").to_string(**fmt)
    logs = REPO_ROOT / "logs"
    d.to_csv(logs / "tonal_diag.csv", index=False)
    (logs / "tonal_diag.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
