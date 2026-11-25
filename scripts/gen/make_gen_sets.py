"""Build the external-noise generalisation sets from the 164 validation clean files only (exp/r2-gen).

Sets (data/gen/<set>/{clean,noisy}/ + samples.csv with the official manifest columns, split = <set>):
  gen_unseen_{dev,test} 328  2 mixtures per clean file, one external noise each (corpus balanced 1/3 each over
                             DEMAND / ESC-50 / MUSAN, SNR balanced over {-5, 0, 5, 10} dB, 82 each).
  gen_multi_{dev,test}  164  two noises from two different corpora (pairs balanced), each normalised to unit
                             power, summed, total SNR 0 dB.
  gen_low               328  the file's own validation noise (noisy - clean) rescaled to -5 dB (_gl0) and
                             -10 dB (_gl1). Test only.
dev sets draw from the gen_dev noise lists with seed 2027, test sets from gen_test with seed 2026
(splits/gen/<corpus>_<split>.txt). Noise files are made active first (scripts/gen/noiselib.py active_noise:
clips with an inactive run >= 0.5 s keep only 20 ms frames above -40 dB re their loudest frame,
concatenated with 5 ms crossfades; nearly silent clips are dropped and counted), then a random excerpt
of the clean file's length is taken, looping if the noise is shorter.
Mixing as scripts/make_val_remix.py: mix = clean + n * sqrt(Pc / (Pn 10^(snr/10))) over the whole file;
joint peak guard (clean and mix scaled together) if |mix| > 0.99; 16-bit PCM, 16 kHz mono, original length.
Run: python -I scripts/gen/make_gen_sets.py [--sets ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "src")]
from noiselib import CORPORA, REPO, active_noise, list_corpus, loop_excerpt, read_noise  # noqa: E402
from nrc.io import EXTRA_SPLITS, load_manifest, read_row, resolve_data_root, to_pcm16  # noqa: E402

SETS = ("gen_unseen_dev", "gen_unseen_test", "gen_multi_dev", "gen_multi_test", "gen_low")
SEEDS = {"gen_dev": 2027, "gen_test": 2026}
UNSEEN_SNRS = (-5.0, 0.0, 5.0, 10.0)
PAIRS = (("demand", "esc50"), ("demand", "musan"), ("esc50", "musan"))
PEAK = 0.99


def load_pool(pool: str, log: list) -> dict[str, list[tuple[str, np.ndarray]]]:
    out, stat_rows = {}, []
    for c in CORPORA:
        rels = (REPO / "splits" / "gen" / f"{c}_{pool}.txt").read_text(encoding="utf-8").split()
        out[c] = []
        cls = {r["rel"]: r["cls"] for r in list_corpus(c)}
        for rel in rels:
            x, status = active_noise(read_noise(rel))
            stat_rows.append({"corpus": c, "rel": rel, "status": status,
                              "seconds_after": 0.0 if x is None else x.size / 16000})
            if x is not None:
                out[c].append((rel, cls[rel], x))
    st = pd.DataFrame(stat_rows)
    st.to_csv(REPO / "logs" / f"gen_activity_{pool}.csv", index=False)
    log.append(f"[{pool}] activity: " + st.groupby(["corpus", "status"]).size().to_string().replace("\n", "; "))
    return out


def mix(c: np.ndarray, n: np.ndarray, snr: float) -> tuple[np.ndarray, np.ndarray, bool]:
    pc, pn = float(np.mean(c ** 2)), float(np.mean(n ** 2))
    assert pc > 0 and pn > 0
    y = c + n * np.sqrt(pc / (pn * 10 ** (snr / 10)))
    pk = float(np.max(np.abs(y)))
    if pk > PEAK:
        return c * (PEAK / pk), y * (PEAK / pk), True
    return c, y, False


def pick(rng, pool, corpus, n):
    rel, cls, x = pool[corpus][int(rng.integers(len(pool[corpus])))]
    off = int(rng.integers(0, x.size - n + 1)) if x.size >= n else int(rng.integers(0, x.size))
    return rel, cls, off, loop_excerpt(x, n, off)


def build(name: str, df: pd.DataFrame, clean: list, noisy_val: list, pools: dict, log: list) -> None:
    out = EXTRA_SPLITS[name]
    for sub in ("clean", "noisy"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("*.wav"):
            old.unlink()
    kind = name.split("_")[1]
    pool_name = "gen_dev" if name.endswith("_dev") else "gen_test"
    rng = np.random.default_rng(SEEDS[pool_name])
    pool = pools.get(pool_name)
    n_clean = len(df)
    jobs = []  # (copy, i, snr, meta, noise)
    if kind == "unseen":
        k = 2
        corp = rng.permutation(np.resize(np.array(CORPORA), k * n_clean))
        snrs = rng.permutation(np.repeat(UNSEEN_SNRS, k * n_clean // len(UNSEEN_SNRS)))
        for copy in range(k):
            for i in range(n_clean):
                j = copy * n_clean + i
                c = str(corp[j])
                rel, cls, off, n = pick(rng, pool, c, clean[i].size)
                jobs.append((copy, i, float(snrs[j]), {"noise_collection": c, "noise_category": cls,
                                                       "noise_file": rel, "noise_offset": off}, n))
    elif kind == "multi":
        pidx = rng.permutation(np.resize(np.arange(len(PAIRS)), n_clean))
        for i in range(n_clean):
            ca, cb = PAIRS[int(pidx[i])]
            ra, ka, oa, na = pick(rng, pool, ca, clean[i].size)
            rb, kb, ob, nb = pick(rng, pool, cb, clean[i].size)
            n = na / np.sqrt(np.mean(na ** 2)) + nb / np.sqrt(np.mean(nb ** 2))
            jobs.append((0, i, 0.0, {"noise_collection": f"{ca}+{cb}", "noise_category": f"{ka}+{kb}",
                                     "noise_file": ra, "noise_offset": oa, "noise_file2": rb,
                                     "noise_offset2": ob}, n))
    else:  # low
        for copy, snr in enumerate((-5.0, -10.0)):
            for i, r in df.iterrows():
                jobs.append((copy, i, snr, {"noise_collection": r.noise_collection,
                                            "noise_category": r.noise_category, "noise_file": "own",
                                            "orig_difficulty": r.noise_difficulty,
                                            "orig_snr_db": float(r.target_snr_db)}, noisy_val[i] - clean[i]))
    tag = {"unseen": "gu", "multi": "gm", "low": "gl"}[kind]
    rows, n_guard = [], 0
    for copy, i, snr, meta, n in jobs:
        r = df.iloc[i]
        c, y, g = mix(clean[i], n, snr)
        n_guard += g
        sid = f"{r.segment_id}_{tag}{copy}"
        crel, nrel = f"clean/{sid}.wav", f"noisy/{sid}_noisy.wav"
        for rel, x in ((crel, c), (nrel, y)):
            pcm, clipped = to_pcm16(x)
            assert clipped == 0, f"{name}/{rel}: {clipped} clipped samples"
            sf.write(str(out / rel), pcm, 16000, subtype="PCM_16")
        code = meta["noise_file"] if kind != "multi" else f"{meta['noise_file']}+{meta['noise_file2']}"
        rows.append({"segment_id": sid, "clean_path": crel, "noisy_path": nrel, "sample_rate_hz": 16000,
                     "num_samples": int(c.size), "noise_category": meta["noise_category"],
                     "noise_collection": meta["noise_collection"],
                     "noise_code": r.noise_code if kind == "low" else Path(code).stem if kind == "unseen" else code,
                     "noise_difficulty": kind, "target_snr_db": snr, "split": name, "clean_id": r.segment_id,
                     **{k2: v for k2, v in meta.items() if k2 not in ("noise_collection", "noise_category")}})
    man = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)
    man.to_csv(out / "samples.csv", index=False, encoding="utf-8")
    load_manifest(resolve_data_root(None), name)  # validate
    # verify: written 16-bit files keep length and the requested SNR
    root = EXTRA_SPLITS[name]
    errs = []
    for _, m in man.iterrows():
        cc, _ = sf.read(str(root / m.clean_path), dtype="float64")
        yy, sr = sf.read(str(root / m.noisy_path), dtype="float64")
        assert sr == 16000 and cc.ndim == yy.ndim == 1 and cc.size == yy.size == m.num_samples
        errs.append(10 * np.log10(np.sum(cc ** 2) / np.sum((yy - cc) ** 2)) - m.target_snr_db)
    errs = np.abs(errs)
    assert errs.max() < 0.05, f"{name}: SNR error {errs.max():.4f} dB"
    log.append(f"{name}: {len(man)} files, peak-guarded {n_guard}, max |SNR err| {errs.max():.4f} dB; "
               f"SNR {man.target_snr_db.value_counts().sort_index().to_dict()}; "
               f"collection {man.noise_collection.value_counts().to_dict()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    args = ap.parse_args(argv)
    root = resolve_data_root(None)
    df = load_manifest(root, "validation").sort_values("segment_id").reset_index(drop=True)
    clean = [read_row(root, r, "clean") for _, r in df.iterrows()]
    noisy = [read_row(root, r, "noisy") for _, r in df.iterrows()]
    log: list[str] = []
    need = {("gen_dev" if s.endswith("_dev") else "gen_test") for s in args.sets if s != "gen_low"}
    pools = {p: load_pool(p, log) for p in sorted(need)}
    for p, pool in pools.items():
        log.append(f"[{p}] usable files: " + str({c: len(v) for c, v in pool.items()}))
    for s in args.sets:
        build(s, df, clean, noisy, pools, log)
    text = "\n".join(log)
    with open(REPO / "logs" / "gen_sets.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
