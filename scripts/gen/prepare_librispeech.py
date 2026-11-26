"""LibriSpeech train-clean-100 subset for ext_mix (exp/r2-ext, gru_v8p2z6a/b): stream, filter, store as int16 shards.

The tarball (data_ext/librispeech/train-clean-100.tar.gz, untrusted) is read as a stream (tarfile "r|gz"); nothing
is extracted, only regular members named LibriSpeech/train-clean-100/<spk>/<chap>/<spk>-<chap>-<utt>.flac are
decoded (soundfile, int16, must be 16 kHz mono).
Pass 1, per utterance: duration, WADA-SNR (scripts/gen/wada.py, Kim & Stern 2008) of the non-zero samples of the
whole utterance (exact int16 zeros, ~1 % of LibriSpeech samples, saturate the estimate at 100 dB), and, for
utterances that pass duration >= 3 s and WADA >= --min-wada, the stored window (at most --max-sec, start drawn from
a generator seeded by crc32(utt id)) is checked against an index of challenge train + validation clean
(Haitsma-Kalker fingerprints, scripts/gen/check_overlap.Index; match = BER < 0.35 over >= 120 frames).
Selection: per speaker the --per-speaker eligible, non-matching utterances with the highest WADA-SNR (ties: id).
Pass 2 writes the selected windows to <out>/shard_XXX.npy + index.csv (nrc.data.extpool.ShardWriter).
Writes data_ext/librispeech/stats.csv (pass 1, reused with --reuse-stats), <out>/prep_summary.json and
logs/r2_ext_librispeech.txt. Run: python -I scripts/gen/prepare_librispeech.py
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
from check_overlap import BER_MAX, Index  # noqa: E402
from noiselib import EXT, REPO, SR, fingerprint  # noqa: E402
from wada import wada_snr  # noqa: E402

from nrc.data.extpool import ShardWriter  # noqa: E402
from nrc.io import load_manifest, read_row, resolve_data_root  # noqa: E402

MEMBER_RE = re.compile(r"^LibriSpeech/train-clean-100/(\d+)/(\d+)/(\1-\2-\d+)\.flac$")
MIN_SEC = 3.0


def members(tar_path: Path):
    """Yield (utt id, speaker, chapter, int16 audio) for every valid FLAC member, in archive order."""
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            mt = MEMBER_RE.match(m.name)
            if not (m.isreg() and mt):
                continue
            x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="int16")
            if sr != SR or x.ndim != 1:
                raise ValueError(f"{m.name}: {sr} Hz, shape {x.shape}")
            yield mt.group(3), mt.group(1), mt.group(2), x


def window(uid: str, n: int, max_len: int) -> int:
    """Start of the stored window (0 if the utterance fits)."""
    if n <= max_len:
        return 0
    return int(np.random.default_rng(zlib.crc32(uid.encode())).integers(n - max_len + 1))


def clean_index() -> Index:
    root = resolve_data_root(None)
    labels, sigs = [], []
    for split in ("train", "validation"):
        for _, r in load_manifest(root, split).iterrows():
            labels.append({"rel": f"{split}/{r.segment_id}"})
            sigs.append(read_row(root, r, "clean"))
    return Index(labels, sigs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tar", default=str(EXT / "librispeech" / "train-clean-100.tar.gz"))
    ap.add_argument("--out", default=str(EXT / "pkg" / "nrc_ext_z6" / "librispeech"))
    ap.add_argument("--min-wada", type=float, default=30.0)
    ap.add_argument("--per-speaker", type=int, default=40)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--reuse-stats", action="store_true")
    args = ap.parse_args(argv)
    tar_path, out = Path(args.tar), Path(args.out)
    stats_path = EXT / "librispeech" / "stats.csv"
    max_len = int(args.max_sec * SR)
    t0 = time.perf_counter()

    if args.reuse_stats and stats_path.is_file():
        st = pd.read_csv(stats_path, dtype={"utt": str, "speaker": str, "chapter": str})
        print(f"pass 1 reused: {stats_path} ({len(st)} utterances)")
    else:
        idx = clean_index()
        print(f"challenge clean index: {len(idx.files)} files, {idx.sub.size} sub-fingerprints", flush=True)
        rows = []
        for k, (uid, spk, chap, x) in enumerate(members(tar_path)):
            xf = x.astype(np.float64) / 32768.0
            dur = x.size / SR
            w = wada_snr(xf[x != 0])  # exact int16 zeros (~1 % of samples) hit the log floor and saturate G
            ber, match = float("nan"), ""
            if dur >= MIN_SEC and np.isfinite(w) and w >= args.min_wada:
                s0 = window(uid, x.size, max_len)
                e, f, m, _ = idx.query(*fingerprint(xf[s0:s0 + max_len]))
                ber = e
                match = idx.files[f]["rel"] if e < BER_MAX else ""
            rows.append({"utt": uid, "speaker": spk, "chapter": chap, "n": x.size, "dur": round(dur, 4),
                         "wada_snr": round(w, 3), "overlap_ber": round(ber, 4), "overlap_match": match})
            if (k + 1) % 2000 == 0:
                print(f"  pass 1: {k + 1} utterances [{time.perf_counter() - t0:.0f}s]", flush=True)
        st = pd.DataFrame(rows)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        st.to_csv(stats_path, index=False)
    st["eligible"] = (st.dur >= MIN_SEC) & (st.wada_snr >= args.min_wada)
    st["match"] = st.overlap_match.fillna("").astype(str).str.len() > 0
    cand = st[st.eligible & ~st.match].sort_values(["speaker", "wada_snr", "utt"], ascending=[True, False, True])
    sel = cand.groupby("speaker", sort=True).head(args.per_speaker)
    keep = set(sel.utt)
    print(f"selected {len(keep)} utterances from {sel.speaker.nunique()} speakers", flush=True)

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; remove it first")
    wr = ShardWriter(out, ["utt", "speaker", "chapter", "wada_snr", "dur_orig", "win_start", "rms"])
    meta = sel.set_index("utt")
    for uid, spk, chap, x in members(tar_path):
        if uid not in keep:
            continue
        s0 = window(uid, x.size, max_len)
        y = np.ascontiguousarray(x[s0:s0 + max_len])
        rms = float(np.sqrt(np.mean((y.astype(np.float64) / 32768.0) ** 2)))
        wr.add(y, utt=uid, speaker=spk, chapter=chap, wada_snr=meta.at[uid, "wada_snr"],
               dur_orig=meta.at[uid, "dur"], win_start=s0, rms=f"{rms:.6g}")
    rows = wr.close()
    if len(rows) != len(keep):
        raise AssertionError(f"wrote {len(rows)} of {len(keep)} selected utterances")
    n_samp = sum(r["length"] for r in rows)
    size = sum(p.stat().st_size for p in out.glob("shard_*.npy"))
    per_spk = sel.groupby("speaker").size()
    summ = {
        "archive": tar_path.name, "utterances_total": int(len(st)), "speakers_total": int(st.speaker.nunique()),
        "hours_total": round(float(st.n.sum()) / SR / 3600, 2),
        "dur_ge_3s": int((st.dur >= MIN_SEC).sum()), "wada_ge_min": int((st.wada_snr >= args.min_wada).sum()),
        "eligible": int(st.eligible.sum()), "overlap_matches": int((st.eligible & st.match).sum()),
        "overlap_ber_min": float(st.overlap_ber.min()) if st.overlap_ber.notna().any() else None,
        "min_wada_db": args.min_wada, "per_speaker_cap": args.per_speaker, "max_sec": args.max_sec,
        "kept_utterances": len(rows), "kept_speakers": int(per_spk.size),
        "speakers_below_cap": int((per_spk < args.per_speaker).sum()),
        "kept_hours": round(n_samp / SR / 3600, 2), "kept_hours_before_crop": round(float(sel.dur.sum()) / 3600, 2),
        "wada_median_all": float(st.wada_snr.median()), "wada_median_kept": float(sel.wada_snr.median()),
        "shards": len(list(out.glob("shard_*.npy"))), "size_bytes": int(size),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "prep_summary.json").write_text(json.dumps(summ, indent=2) + "\n", encoding="utf-8")
    text = "LibriSpeech train-clean-100 subset (scripts/gen/prepare_librispeech.py)\n" + json.dumps(summ, indent=2)
    (REPO / "logs" / "r2_ext_librispeech.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
