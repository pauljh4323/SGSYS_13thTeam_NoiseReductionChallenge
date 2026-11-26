"""Content overlap check between the challenge noise and the external corpora, and within/among the corpora.

Index: Haitsma-Kalker style fingerprints (scripts/gen/noiselib.py) of every DEMAND ch01 / ESC-50 / MUSAN-noise
file. Queries:
  A. challenge noise estimates n = noisy - clean of every train + validation file (1460; analysis only, never
     used by any method),
  B. every corpus file in 3 s windows (hop 3 s, at most 20 windows per file) against all OTHER corpus
     recordings (other file and, for ESC-50, other src_file).
Candidate alignments come from exact sub-fingerprint hits (sub-fingerprints with > 200 index hits ignored),
the top 5 by votes are verified by bit error rate (BER) over the aligned block; a match is BER < 0.35 over
>= 120 valid frames (~2 s). Positive control: 40 random corpus windows with -6 dB gain + white noise at
15 dB SNR must match their own file. Writes splits/gen/overlap_matches.csv, splits/gen/excluded.txt
(both members of every corpus-corpus match, and every corpus file matched by challenge noise) and
logs/gen_overlap.txt. Run: python -I scripts/gen/check_overlap.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "src")]
from noiselib import CORPORA, REPO, SR, ber, fingerprint, list_corpus, read_noise  # noqa: E402
from nrc.io import load_manifest, read_row, resolve_data_root  # noqa: E402

BER_MAX, MIN_FRAMES, MAX_HITS, TOP = 0.35, 120, 200, 5
WIN, MAX_WIN = 3 * SR, 20


class Index:
    def __init__(self, files: list[dict], signals: list[np.ndarray] | None = None):
        """files: [{rel, ...}] read with read_noise, or any labels when `signals` (same order) is given."""
        self.files, self.fps = files, []
        subs, fid, pos = [], [], []
        for i, f in enumerate(files):
            s, v = fingerprint(read_noise(f["rel"]) if signals is None else signals[i])
            self.fps.append((s, v))
            k = np.flatnonzero(v)
            subs.append(s[k]); fid.append(np.full(k.size, i, np.int32)); pos.append(k.astype(np.int64))
        sub, fid, pos = np.concatenate(subs), np.concatenate(fid), np.concatenate(pos)
        o = np.argsort(sub, kind="stable")
        self.sub, self.fid, self.pos = sub[o], fid[o], pos[o]

    def query(self, q: np.ndarray, qv: np.ndarray, skip=lambda i: False) -> tuple[float, int, int, int]:
        """-> (best BER, file index, n frames compared, offset) over the top candidate alignments."""
        votes = Counter()
        for t in np.flatnonzero(qv):
            lo, hi = np.searchsorted(self.sub, q[t], "left"), np.searchsorted(self.sub, q[t], "right")
            if 0 < hi - lo <= MAX_HITS:
                for f, p in zip(self.fid[lo:hi], self.pos[lo:hi]):
                    if not skip(int(f)):
                        votes[(int(f), int(p) - int(t))] += 1
        best = (1.0, -1, 0, 0)
        for (f, off), _ in votes.most_common(TOP):
            r, rv = self.fps[f]
            a, b = max(0, -off), min(q.size, r.size - off)
            if b - a <= 0:
                continue
            m = int((qv[a:b] & rv[a + off:b + off]).sum())
            e = ber(q[a:b], qv[a:b], r[a + off:b + off], rv[a + off:b + off])
            if m >= MIN_FRAMES and e < best[0]:
                best = (e, f, m, off)
        return best


def windows(x: np.ndarray):
    starts = range(0, max(1, x.size - WIN + 1), WIN)
    return [x[s:s + WIN] for s in list(starts)[:MAX_WIN]]


def main() -> int:
    files = [f for c in CORPORA for f in list_corpus(c)]
    log = [f"index: {len(files)} files " + str(Counter(f['corpus'] for f in files))]
    idx = Index(files)
    log.append(f"index sub-fingerprints: {idx.sub.size}")
    rows = []

    # positive control
    rng = np.random.default_rng(0)
    ok = 0
    ctrl = rng.choice(len(files), 40, replace=False)
    for i in ctrl:
        x = read_noise(files[i]["rel"])
        s0 = int(rng.integers(0, max(1, x.size - WIN)))
        w = x[s0:s0 + WIN] * 10 ** (-6 / 20)
        pw = np.mean(w ** 2)
        w = w + rng.standard_normal(w.size) * np.sqrt(pw / 10 ** 1.5)
        e, f, m, _ = idx.query(*fingerprint(w))
        ok += int(f == i and e < BER_MAX)
    log.append(f"positive control: {ok}/40 matched own file (BER < {BER_MAX})")

    # A. challenge noise
    root = resolve_data_root(None)
    best_a = []
    for split in ("train", "validation"):
        df = load_manifest(root, split)
        for _, r in df.iterrows():
            n = read_row(root, r, "noisy") - read_row(root, r, "clean")
            e, f, m, off = idx.query(*fingerprint(n))
            best_a.append(e)
            if e < BER_MAX:
                rows.append({"query": f"{split}/{r.segment_id}", "query_kind": "challenge",
                             "query_meta": f"{r.noise_collection}|{r.noise_category}|{r.noise_code}",
                             "match": files[f]["rel"], "ber": e, "frames": m, "offset": off})
    log.append(f"A challenge queries: {len(best_a)}; matches {sum(x < BER_MAX for x in best_a)}; "
               f"best BER min {min(best_a):.3f} / 1st pct {np.percentile(best_a, 1):.3f} / median "
               f"{np.median(best_a):.3f}")

    # B. corpus vs other recordings
    best_b, nq = [], 0
    for i, fq in enumerate(files):
        x = read_noise(fq["rel"])
        same = (lambda j, i=i, fq=fq: j == i or (fq["corpus"] == "esc50" and files[j]["corpus"] == "esc50"
                                                 and files[j]["group"] == fq["group"]))
        for w in windows(x):
            nq += 1
            e, f, m, off = idx.query(*fingerprint(w), skip=same)
            best_b.append(e)
            if e < BER_MAX:
                rows.append({"query": fq["rel"], "query_kind": "corpus", "query_meta": fq["corpus"],
                             "match": files[f]["rel"], "ber": e, "frames": m, "offset": off})
    log.append(f"B corpus window queries: {nq}; matching windows {sum(x < BER_MAX for x in best_b)}; "
               f"best BER 1st pct {np.percentile(best_b, 1):.3f} / median {np.median(best_b):.3f}")

    out = REPO / "splits" / "gen"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows, columns=["query", "query_kind", "query_meta", "match", "ber", "frames", "offset"])
    m.to_csv(out / "overlap_matches.csv", index=False, float_format="%.4f")
    excl = set(m["match"]) | set(m.loc[m.query_kind == "corpus", "query"])
    (out / "excluded.txt").write_text("".join(f"{e}\n" for e in sorted(excl)), encoding="utf-8")
    log.append(f"excluded corpus files: {len(excl)}")
    if len(m):
        log.append(m.groupby(["query_kind", "query_meta"]).size().to_string())
    text = "\n".join(log)
    (REPO / "logs" / "gen_overlap.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
