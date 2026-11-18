"""Round-2 diagnosis (analysis only; branch exp/r2-diag): noise-type overlap, per-category scores of
gru_v5b_off vs dsp_p (dsp-v8), per-category routing / blending bounds, noise-category identifiability.

Stages (each caches its result under outputs/r2_diag/, so stages can be re-run independently):
  enhance   run gru_v5b_off and dsp_p on validation + val_remix, write 16-bit wavs exactly as main.py
  blend     y = w*gru + (1-w)*dsp (w in 0, .25, .5, .75, 1) from the 16-bit outputs, re-quantised to 16 bit,
            scored with scripts/eval.py's own evaluate_row (source "pred"); per-file CSVs
  features  dsp-v8 stationary noise PSD (spp_mean, dsp_p params) pooled to 32 log bands + temporal variance
            of the log band energy of the noisy STFT, for train / validation / val_remix noisy files
  report    overlap, per-category tables, routing bounds, classifiers -> logs/r2_diag_tables.md

Methods only ever see the noisy waveform; clean audio is read by evaluate_row for scoring only.
Blend scores are written to outputs/r2_diag/ and logs/ only (not to results/experiments.csv): they are
analysis bounds, not candidate methods.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT, load_config

from nrc.io import load_manifest, read_row, read_wav, resolve_data_root, split_root, to_pcm16, write_wav_pcm16
from nrc.methods import get_method
from nrc.methods import mmse_lsa
from nrc.stft import stft

OUT = REPO_ROOT / "outputs" / "r2_diag"
SPLITS = ("validation", "val_remix")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CFG = {"gru": REPO_ROOT / "configs" / "dl" / "gru_v5b_off.yaml", "dsp": REPO_ROOT / "configs" / "dsp_p.yaml"}
N_BANDS = 32


def _eval_module():
    spec = importlib.util.spec_from_file_location("nrc_eval", REPO_ROOT / "scripts" / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EVAL = None


def _score_task(task):
    """scripts/eval.py evaluate_row, loaded once per worker process (picklable top-level wrapper)."""
    global _EVAL
    if _EVAL is None:
        _EVAL = _eval_module()
    return _EVAL.evaluate_row(task)


def wav_name(row) -> str:
    return Path(row["noisy_path"]).name


# ---------------------------------------------------------------- enhance
def _enhance_task(task):
    root, row, method, params, out_path = task
    if out_path.exists():
        return 0
    noisy = read_row(root, row, "noisy")
    y = get_method(method, **params)(noisy, int(row["sample_rate_hz"]))
    return write_wav_pcm16(out_path, y, int(row["sample_rate_hz"]))


def stage_enhance(data_root, jobs):
    for key, cfg_path in CFG.items():
        cfg = load_config(cfg_path)
        for split in SPLITS:
            man = load_manifest(data_root, split)
            root = split_root(data_root, split)
            d = OUT / split / key
            d.mkdir(parents=True, exist_ok=True)
            tasks = [(root, r.to_dict(), cfg["method"], cfg["params"], d / wav_name(r))
                     for _, r in man.iterrows()]
            # gru: torch already multi-threads inside one process; keep it to 2 workers
            n = min(jobs, 2) if key == "gru" else jobs
            with ProcessPoolExecutor(max_workers=n) as pool:
                clipped = sum(pool.map(_enhance_task, tasks, chunksize=4))
            print(f"enhance {key} {split}: {len(tasks)} files, clipped samples {clipped}", flush=True)


# ---------------------------------------------------------------- blend + score
def _blend_write(task):
    gpath, dpath, out_path, w, sr, n = task
    if out_path.exists():
        return 0
    g, d = read_wav(gpath, sr, n), read_wav(dpath, sr, n)
    return write_wav_pcm16(out_path, w * g + (1.0 - w) * d, sr)


def stage_blend(data_root, jobs):
    cfg = load_config(CFG["gru"])  # only used for its name inside evaluate_row (source "pred")
    for split in SPLITS:
        man = load_manifest(data_root, split)
        root = split_root(data_root, split)
        for w in WEIGHTS:
            csv = OUT / f"score_{split}_w{w:g}.csv"
            if csv.exists():
                continue
            d = OUT / split / f"w{w:g}"
            d.mkdir(parents=True, exist_ok=True)
            bt = [(OUT / split / "gru" / wav_name(r), OUT / split / "dsp" / wav_name(r), d / wav_name(r), w,
                   int(r["sample_rate_hz"]), int(r["num_samples"])) for _, r in man.iterrows()]
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                list(pool.map(_blend_write, bt, chunksize=8))
            tasks = [(root, r.to_dict(), cfg, split, "pred", d / wav_name(r), 0.0, 0.0)
                     for _, r in man.iterrows()]
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                rows = list(pool.map(_score_task, tasks, chunksize=4))
            df = pd.DataFrame(rows)
            df.to_csv(csv, index=False)
            print(f"blend {split} w={w:g}: PESQ {df.pesq.mean():.4f} SNR {df.snr_db.mean():.3f} "
                  f"failed {int(df.pesq.isna().sum())}", flush=True)


# ---------------------------------------------------------------- features
def band_edges(n_fft, sr, n_bands=N_BANDS, f_lo=50.0):
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(f_lo, sr / 2, n_bands + 1)
    idx = np.searchsorted(freqs, edges)
    idx[0] = 1  # skip DC
    idx[-1] = len(freqs)
    for i in range(1, len(idx)):  # every band gets >= 1 bin
        idx[i] = max(idx[i], idx[i - 1] + 1)
    return idx


def _feature_task(task):
    root, row, p = task
    x = read_row(root, row, "noisy")
    sr = int(row["sample_rate_hz"])
    n_fft, hop = int(p["n_fft"]), int(p["hop"])
    _, times, z = stft(x, sr, n_fft, hop, p["window"])
    power = np.abs(z) ** 2
    est = mmse_lsa.stationary_noise_psd(z, power, times, len(x), sr, n_fft, "spp_mean",
                                        p["noise_fraction"], spp_iters=p["spp_iters"],
                                        spp_xi_h1_db=p["spp_xi_h1_db"])
    idx = band_edges(n_fft, sr)
    eps = 1e-12
    psd_b = np.array([est[a:b].mean() for a, b in zip(idx[:-1], idx[1:])])
    eng_b = np.stack([power[a:b].mean(axis=0) for a, b in zip(idx[:-1], idx[1:])])  # (bands, frames)
    var_b = np.var(10 * np.log10(eng_b + eps), axis=1)
    return np.concatenate([10 * np.log10(psd_b + eps), var_b])


def stage_features(data_root, jobs):
    p = load_config(CFG["dsp"])["params"]
    for split in ("train",) + SPLITS:
        path = OUT / f"feat_{split}.npz"
        if path.exists():
            continue
        man = load_manifest(data_root, split)
        root = split_root(data_root, split)
        tasks = [(root, r.to_dict(), p) for _, r in man.iterrows()]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            feats = np.stack(list(pool.map(_feature_task, tasks, chunksize=8)))
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=feats, seg=man.segment_id.to_numpy(), cat=man.noise_category.to_numpy())
        print(f"features {split}: {feats.shape}", flush=True)


# ---------------------------------------------------------------- report helpers
def md(df: pd.DataFrame, floatfmt=".4f") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load_scores(split, man):
    """Per-file table: noisy / dsp / gru / blend PESQ + SNR, joined with manifest metadata."""
    base = man[["segment_id", "noise_category", "noise_collection", "noise_code", "noise_difficulty",
                "target_snr_db"]].copy()
    for w in WEIGHTS:
        s = pd.read_csv(OUT / f"score_{split}_w{w:g}.csv").set_index("segment_id")
        base[f"pesq_w{w:g}"] = base.segment_id.map(s.pesq)
        base[f"snr_w{w:g}"] = base.segment_id.map(s.snr_db)
        if w == 0.0:
            base["pesq_noisy"] = base.segment_id.map(s.pesq - s.delta_pesq)
            base["snr_noisy"] = base.segment_id.map(s.snr_db - s.delta_snr_db)
    base["dsp"], base["gru"], base["noisy"] = base["pesq_w0"], base["pesq_w1"], base["pesq_noisy"]
    return base


def per_group(df, key):
    g = df.groupby(key)
    t = pd.DataFrame({"n": g.size(), "noisy": g.noisy.mean(), "dsp_v8": g.dsp.mean(), "gru_v5b": g.gru.mean(),
                      "gru-dsp": g.apply(lambda s: (s.gru - s.dsp).mean()),
                      "gru gain": g.apply(lambda s: (s.gru - s.noisy).mean()),
                      "dsp gain": g.apply(lambda s: (s.dsp - s.noisy).mean()),
                      "gru>dsp files": g.apply(lambda s: f"{int((s.gru > s.dsp).sum())}/{len(s)}"),
                      "SNR dsp": g.snr_w0.mean(), "SNR gru": g.snr_w1.mean()})
    return t.reset_index()


def knn_predict(xtr, ytr, xte, k=5):
    d = ((xte[:, None, :] - xtr[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    classes = np.unique(ytr)
    scores = np.zeros((len(xte), len(classes)))
    for j in range(k):  # vote, ties broken by summed inverse rank
        lab = ytr[nn[:, j]]
        ci = np.searchsorted(classes, lab)
        scores[np.arange(len(xte)), ci] += 1.0 + 1e-3 * (k - j)
    return classes, scores


def centroid_predict(xtr, ytr, xte):
    classes = np.unique(ytr)
    c = np.stack([xtr[ytr == k].mean(0) for k in classes])
    d = ((xte[:, None, :] - c[None]) ** 2).sum(-1)
    return classes, -d


def topk(classes, scores, y, k):
    order = np.argsort(-scores, axis=1)[:, :k]
    return np.array([y[i] in classes[order[i]] for i in range(len(y))])


# ---------------------------------------------------------------- report
def stage_report(data_root):
    out = []
    man = {s: load_manifest(data_root, s) for s in ("train",) + SPLITS}
    tr, va, vr = man["train"], man["validation"], man["val_remix"]

    # 1. overlap
    out.append("## 1. Split overlap\n")
    rows = []
    for col in ("noise_code", "noise_collection"):
        a, b, c = set(tr[col]), set(va[col]), set(vr[col])
        rows.append({"field": col, "train": len(a), "validation": len(b), "train only": len(a - b),
                     "validation only": len(b - a), "both": len(a & b),
                     "val_remix codes in train": f"{len(c & a)}/{len(c)}"})
    out.append(md(pd.DataFrame(rows)))
    cat = pd.DataFrame({"train files": tr.noise_category.value_counts(),
                        "validation files": va.noise_category.value_counts(),
                        "val_remix files": vr.noise_category.value_counts(),
                        "train codes": tr.groupby("noise_category").noise_code.nunique(),
                        "validation codes": va.groupby("noise_category").noise_code.nunique()})
    shared = {k: len(set(tr[tr.noise_category == k].noise_code) & set(va[va.noise_category == k].noise_code))
              for k in cat.index}
    cat["val codes also in train"] = pd.Series(shared)
    cat["collections"] = pd.concat([tr, va]).groupby("noise_category").noise_collection.agg(
        lambda s: "/".join(sorted(set(s))))
    out.append("\n" + md(cat.fillna(0).reset_index(names="noise_category"), ".0f"))
    vcodes = va.noise_code.isin(set(tr.noise_code))
    out.append(f"\nValidation files whose noise_code also occurs in train: {int(vcodes.sum())}/{len(va)}; "
               f"val_remix: {int(vr.noise_code.isin(set(tr.noise_code)).sum())}/{len(vr)}.")
    # val_remix noise source is a validation file
    if "noise_source_id" in vr.columns:
        src_in_val = vr.noise_source_id.isin(set(va.segment_id)).mean()
        out.append(f"val_remix noise_source_id taken from validation segments: {src_in_val:.0%}.")
    # code-level table
    codes = pd.concat([tr.assign(split="train"), va.assign(split="validation")])
    ct = codes.groupby(["noise_category", "noise_collection", "noise_code"]).split.agg(
        lambda s: "both" if s.nunique() == 2 else s.iloc[0]).reset_index()
    out.append("\n<details><summary>noise_code membership (train / validation)</summary>\n\n"
               + md(ct) + "\n\n</details>")

    # 2. per-file scores
    sc = {s: load_scores(s, man[s]) for s in SPLITS}
    for s in SPLITS:
        out.append(f"\n## 2. Per-category scores, {s}\n")
        o = sc[s]
        out.append(f"Overall n={len(o)}: noisy {o.noisy.mean():.4f}, dsp_v8 {o.dsp.mean():.4f} "
                   f"(SNR {o.snr_w0.mean():.2f}), gru_v5b {o.gru.mean():.4f} (SNR {o.snr_w1.mean():.2f}), "
                   f"gru-dsp {(o.gru - o.dsp).mean():+.4f}, gru>dsp files {int((o.gru > o.dsp).sum())}/{len(o)}\n")
        t = per_group(o, "noise_category").sort_values("gru_v5b")
        out.append(md(t))
        out.append("\n" + md(per_group(o, "noise_difficulty")))
        out.append("\n" + md(per_group(o, "target_snr_db")))
        out.append("\nWeakest 5 for gru_v5b by PESQ: " + ", ".join(
            f"{r.noise_category} (n {r.n}, {r.gru_v5b:.3f}, gain {r['gru gain']:+.3f}, gru-dsp {r['gru-dsp']:+.3f})"
            for _, r in t.head(5).iterrows()))
        t2 = t.sort_values("gru gain")
        out.append("\nWeakest 5 for gru_v5b by gain over noisy: " + ", ".join(
            f"{r.noise_category} (n {r.n}, {r.gru_v5b:.3f}, gain {r['gru gain']:+.3f}, gru-dsp {r['gru-dsp']:+.3f})"
            for _, r in t2.head(5).iterrows()))

    # 3. routing bounds
    out.append("\n## 3. Routing / blending bounds\n")
    rows = []
    v, r = sc["validation"], sc["val_remix"]
    wcols = [f"pesq_w{w:g}" for w in WEIGHTS]
    for name, o in sc.items():
        rows.append({"split": name, "gru_v5b": o.gru.mean(), "dsp_v8": o.dsp.mean(),
                     "(a) oracle max(gru,dsp)": np.maximum(o.gru, o.dsp).mean(),
                     "oracle files dsp": int((o.dsp > o.gru).sum()),
                     "oracle max over 5 w": o[wcols].max(axis=1).mean(),
                     **{f"global w={w:g}": o[f"pesq_w{w:g}"].mean() for w in WEIGHTS}})
    out.append(md(pd.DataFrame(rows)))
    # (b) per-category hard routing chosen on validation
    vm = v.groupby("noise_category")[["gru", "dsp"]].mean()
    choice = (vm.dsp > vm.gru).map({True: "dsp", False: "gru"})
    # (c) per-category w chosen on validation
    vw = v.groupby("noise_category")[wcols].mean()
    wbest = vw.idxmax(axis=1).str.replace("pesq_w", "").astype(float)
    res = []
    for name, o in sc.items():
        rb = np.where(o.noise_category.map(choice).fillna("gru") == "dsp", o.dsp, o.gru)
        wc = o.noise_category.map(wbest).fillna(1.0)
        rc = np.array([o.loc[i, f"pesq_w{w:g}"] for i, w in zip(o.index, wc)])
        # in-sample oracle per category on this split (upper bound of the category rule)
        own = o.groupby("noise_category")[wcols].mean().max(axis=1)
        own_n = o.groupby("noise_category").size()
        res.append({"split": name, "gru_v5b": o.gru.mean(),
                    "(b) route rule (from val)": rb.mean(), "(b) gain": rb.mean() - o.gru.mean(),
                    "(c) blend rule (from val)": rc.mean(), "(c) gain": rc.mean() - o.gru.mean(),
                    "(c) in-split oracle w per cat": (own * own_n).sum() / own_n.sum(),
                    "(c) oracle gain": (own * own_n).sum() / own_n.sum() - o.gru.mean()})
    out.append("\n" + md(pd.DataFrame(res)))
    rule = pd.DataFrame({"val gru": vm.gru, "val dsp": vm.dsp, "(b) choice": choice, "(c) w": wbest,
                         "val best-w PESQ": vw.max(axis=1),
                         "val_remix gru": r.groupby("noise_category").gru.mean(),
                         "val_remix at rule w": r.assign(x=[r.loc[i, f"pesq_w{w:g}"] for i, w in
                                                         zip(r.index, r.noise_category.map(wbest).fillna(1.0))])
                         .groupby("noise_category").x.mean()}).reset_index(names="noise_category")
    out.append("\n" + md(rule))
    # cross-split check of the rule direction: also choose on val_remix, apply to validation
    rm_ = r.groupby("noise_category")[wcols].mean()
    wbest_r = rm_.idxmax(axis=1).str.replace("pesq_w", "").astype(float)
    vc = np.array([v.loc[i, f"pesq_w{w:g}"] for i, w in zip(v.index, v.noise_category.map(wbest_r).fillna(1.0))])
    out.append(f"\nReverse direction (w per category chosen on val_remix, applied to validation): "
               f"{vc.mean():.4f} (gain {vc.mean() - v.gru.mean():+.4f}); categories where the two splits pick "
               f"the same w: {int((wbest == wbest_r.reindex(wbest.index)).sum())}/{len(wbest)}; "
               f"Spearman of per-category gru-dsp delta val vs val_remix: "
               f"{(vm.gru - vm.dsp).rank().corr((r.groupby('noise_category').gru.mean() - r.groupby('noise_category').dsp.mean()).rank()):.3f}.")

    # 4. identifiability
    out.append("\n## 4. Category identifiability from noisy input\n")
    F = {s: np.load(OUT / f"feat_{s}.npz", allow_pickle=True) for s in ("train",) + SPLITS}
    variants = {"logPSD+var (64-d)": lambda x: x,
                "level-normalised logPSD+var": lambda x: np.concatenate(
                    [x[:, :N_BANDS] - x[:, :N_BANDS].mean(1, keepdims=True), x[:, N_BANDS:]], 1),
                "logPSD only (32-d)": lambda x: x[:, :N_BANDS]}
    acc_rows, per_cat, conf_txt = [], {}, []
    for vname, fx in variants.items():
        xtr = fx(F["train"]["x"])
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
        xtr = (xtr - mu) / sd
        ytr = F["train"]["cat"]
        for s in SPLITS:
            xte, yte = (fx(F[s]["x"]) - mu) / sd, F[s]["cat"]
            for cname, fn in (("nearest centroid", lambda: centroid_predict(xtr, ytr, xte)),
                              ("k-NN k=5", lambda: knn_predict(xtr, ytr, xte, 5))):
                classes, scores = fn()
                t1, t3 = topk(classes, scores, yte, 1), topk(classes, scores, yte, 3)
                acc_rows.append({"features": vname, "classifier": cname, "split": s,
                                 "top-1": t1.mean(), "top-3": t3.mean()})
                if vname == "logPSD+var (64-d)":
                    per_cat[(cname, s)] = pd.DataFrame({"y": yte, "t1": t1, "t3": t3}).groupby("y").agg(
                        n=("t1", "size"), top1=("t1", "mean"), top3=("t3", "mean"))
                    pred = classes[np.argmax(scores, 1)]
                    cm = pd.crosstab(pd.Series(yte, name="true"), pd.Series(pred, name="pred"))
                    off = [(a, b, int(cm.loc[a, b])) for a in cm.index for b in cm.columns
                           if a != b and cm.loc[a, b] > 0]
                    off.sort(key=lambda z: -z[2])
                    conf_txt.append(f"- {cname}, {s}: top confusions (true -> pred, files): " +
                                    "; ".join(f"{a} -> {b} {c}" for a, b, c in off[:8]))
    # train leave-one-code-out sanity is not needed; report chance level
    n_cls = len(np.unique(F["train"]["cat"]))
    out.append(f"Classes: {n_cls} noise categories (chance top-1 {1 / n_cls:.3f}, top-3 {3 / n_cls:.3f}); "
               f"features z-scored with train statistics; train = {len(F['train']['x'])} noisy files.\n")
    out.append(md(pd.DataFrame(acc_rows), ".3f"))
    pc = pd.concat({f"{c} / {s}": t for (c, s), t in per_cat.items()}, axis=1)
    pc.columns = [f"{a} {b}" for a, b in pc.columns]
    pc = pc.loc[:, ~pc.columns.str.endswith(" n") | (pc.columns == pc.columns[0])]
    out.append("\nPer category (logPSD+var features):\n\n" + md(pc.reset_index(names="category"), ".2f"))
    out.append("\nConfusion summary (logPSD+var):\n\n" + "\n".join(conf_txt))

    # also: does routing depend on correct identification? classifier-driven routing on val_remix
    xtr = F["train"]["x"]
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    xtr = (xtr - mu) / sd
    ytr = F["train"]["cat"]
    rows = []
    for s in SPLITS:
        xte = (F[s]["x"] - mu) / sd
        classes, scores = knn_predict(xtr, ytr, xte, 5)
        pred = pd.Series(classes[np.argmax(scores, 1)], index=F[s]["seg"])
        o = sc[s].set_index("segment_id")
        wpred = pred.reindex(o.index).map(wbest).fillna(1.0)
        y = np.array([o.loc[i, f"pesq_w{w:g}"] for i, w in zip(o.index, wpred)])
        rows.append({"split": s, "gru_v5b": o.gru.mean(), "(c) rule via k-NN predicted category": y.mean(),
                     "gain": y.mean() - o.gru.mean()})
    out.append("\nBlend rule (c) driven by the k-NN predicted category instead of the true label "
               "(validation row is in-sample for the rule):\n\n" + md(pd.DataFrame(rows)))

    text = "\n".join(out) + "\n"
    (REPO_ROOT / "logs").mkdir(exist_ok=True)
    (REPO_ROOT / "logs" / "r2_diag_tables.md").write_text(text, encoding="utf-8")
    for s in SPLITS:
        sc[s].to_csv(OUT / f"perfile_{s}.csv", index=False)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=("enhance", "blend", "features", "report"))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args(argv)
    data_root = resolve_data_root(a.data_root)
    for st in a.stages:
        {"enhance": lambda: stage_enhance(data_root, a.jobs), "blend": lambda: stage_blend(data_root, a.jobs),
         "features": lambda: stage_features(data_root, a.jobs), "report": lambda: stage_report(data_root)}[st]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
