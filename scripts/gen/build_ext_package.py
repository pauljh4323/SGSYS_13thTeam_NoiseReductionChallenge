"""Build the ext_mix Drive package data_ext/pkg/nrc_ext_z6 (exp/r2-ext, gru_v8p2z6a/b).

Expects <pkg>/librispeech from scripts/gen/prepare_librispeech.py, then adds
  <pkg>/noise_pool/   int16 shards + index.csv of the external noise **train_pool only**
                      (splits/gen/{demand,esc50,musan}_train_pool.txt; excluded.txt removed; noiselib.active_noise
                      applied, nearly silent clips dropped; status per file in index.csv)
  <pkg>/gen/          copies of data/gen/gen_unseen_dev and data/gen/gen_multi_dev (selection sets)
  <pkg>/MANIFEST.json SHA-256 and size of every file, split-list hashes, builder git hash, totals
The noise list is checked with nrc.data.extpool.check_noise_rels (no gen_dev / gen_test / excluded file) before
anything is written, and the written index is checked again. Run: python -I scripts/gen/build_ext_package.py
--verify only re-hashes an existing package against its MANIFEST.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
from noiselib import EXT, REPO, SR, active_noise, read_noise  # noqa: E402

from nrc.data.extpool import CORPORA, SPLITS_DIR, ExtPool, ShardWriter, check_noise_rels, split_sets  # noqa: E402

PKG = EXT / "pkg" / "nrc_ext_z6"
DEV_SETS = ("gen_unseen_dev", "gen_multi_dev")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def verify(pkg: Path) -> int:
    man = json.loads((pkg / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [rel for rel, (h, n) in man["files"].items()
           if not (pkg / rel).is_file() or (pkg / rel).stat().st_size != n or sha256(pkg / rel) != h]
    print(f"verify {pkg}: {len(man['files'])} files, {man['total_bytes'] / 1e9:.3f} GB, {len(bad)} bad")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkg", default=str(PKG))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    pkg = Path(args.pkg)
    if args.verify:
        return verify(pkg)
    if not (pkg / "librispeech" / "index.csv").is_file():
        raise FileNotFoundError("run scripts/gen/prepare_librispeech.py first")

    # noise train_pool
    excluded = {ln.strip() for ln in (SPLITS_DIR / "excluded.txt").read_text(encoding="utf-8").splitlines()}
    rels = []
    for c in CORPORA:
        lst = [ln.strip() for ln in (SPLITS_DIR / f"{c}_train_pool.txt").read_text(encoding="utf-8").splitlines()
               if ln.strip()]
        rels += [r for r in lst if r not in excluded]
    check_noise_rels(rels)
    nd = pkg / "noise_pool"
    if nd.exists():
        shutil.rmtree(nd)
    wr = ShardWriter(nd, ["corpus", "rel", "status", "orig_len"])
    status = Counter()
    for rel in rels:
        x = read_noise(rel)
        y, st = active_noise(x)
        status[(rel.split("/")[0], st)] += 1
        if y is None:
            continue
        wr.add(to_int16(y), corpus=rel.split("/")[0], rel=rel, status=st, orig_len=x.size)
    rows = wr.close()

    # dev sets
    gd = pkg / "gen"
    for s in DEV_SETS:
        if (gd / s).exists():
            shutil.rmtree(gd / s)
        shutil.copytree(REPO / "data" / "gen" / s, gd / s)

    pool = ExtPool(pkg)  # re-checks the written noise index against the split lists
    allowed, forbidden = split_sets()
    assert not ({r["rel"] for r in pool.noise.rows} & forbidden)

    files = {}
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files[p.relative_to(pkg).as_posix()] = [sha256(p), p.stat().st_size]
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    noise_s = {c: round(sum(r["length"] for r in rows if r["corpus"] == c) / SR / 3600, 3) for c in CORPORA}
    groups = {k: sum(v for (_, kk), v in status.items() if kk == k) for k in ("kept", "trimmed", "silent")}
    man = {
        "package": "nrc_ext_z6", "builder_git": git,
        "split_lists_sha256": {p.name: sha256(p) for p in sorted(SPLITS_DIR.glob("*.txt"))},
        "noise_files": len(rows), "noise_hours": noise_s, "noise_status": groups,
        "noise_per_corpus": {c: sum(1 for r in rows if r["corpus"] == c) for c in CORPORA},
        "librispeech": pool.summary(),
        "total_bytes": sum(n for _, n in files.values()), "files": files,
    }
    (pkg / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    size = {d: sum(n for rel, (_, n) in files.items() if rel.startswith(d + "/")) for d in ("librispeech", "noise_pool",
                                                                                         "gen")}
    text = (f"package {pkg}\n{pool.summary()}\nnoise status {groups}; per corpus {man['noise_per_corpus']}\n"
            f"sizes (GB): " + ", ".join(f"{k} {v / 1e9:.3f}" for k, v in size.items())
            + f"; total {man['total_bytes'] / 1e9:.3f} GB ({man['total_bytes'] / 2**30:.3f} GiB), {len(files)} files")
    (REPO / "logs" / "r2_ext_package.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
