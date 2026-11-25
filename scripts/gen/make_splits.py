"""Deterministic, disjoint recording-level splits of the external corpora: train_pool 70 % / gen_dev 10 % /
gen_test 20 % (exp/r2-gen).

  DEMAND  by environment: environments sorted by sha1(name); first round(20 %) -> gen_test, next round(10 %)
          -> gen_dev, rest -> train_pool (17 environments at 16 kHz: 3 / 2 / 12).
  ESC-50  by fold: fold 5 -> gen_test, folds 1-3 -> train_pool, fold 4 split by freesound source id
          (sha1(src_file) parity: even -> train_pool, odd -> gen_dev), so a source recording never spans splits.
          The fold used is the source's highest fold (4 sources have clips in two folds).
  MUSAN   by file: sha1(relative path) mod 100: < 70 train_pool, < 80 gen_dev, else gen_test.
Files in splits/gen/excluded.txt (scripts/gen/check_overlap.py) are left out of every split.
Writes splits/gen/<corpus>_<split>.txt (paths relative to data_ext/) and splits/gen/summary.csv.
Run: python -I scripts/gen/make_splits.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noiselib import CORPORA, REPO, list_corpus  # noqa: E402

SPLITS = ("train_pool", "gen_dev", "gen_test")
OUT = REPO / "splits" / "gen"


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)


def assign(corpus: str, rows: list[dict]) -> dict[str, str]:
    """rel -> split."""
    if corpus == "demand":
        envs = sorted({r["group"] for r in rows}, key=h)
        n_test, n_dev = round(0.2 * len(envs)), round(0.1 * len(envs))
        env_split = {e: "gen_test" if i < n_test else "gen_dev" if i < n_test + n_dev else "train_pool"
                     for i, e in enumerate(envs)}
        return {r["rel"]: env_split[r["group"]] for r in rows}
    if corpus == "esc50":
        # 4 of 1524 freesound sources have clips in two folds (2/3 and 4/5): decide per source by its highest
        # fold so a source recording never spans splits (moves 4-209698-A-37 and 4-234879-A-6 to gen_test).
        top = {}
        for r in rows:
            top[r["group"]] = max(top.get(r["group"], 0), int(r["fold"]))

        def one(r):
            fold = top[r["group"]]
            if fold == 5:
                return "gen_test"
            if fold == 4:
                return "gen_dev" if h(r["group"]) % 2 else "train_pool"
            return "train_pool"
        return {r["rel"]: one(r) for r in rows}
    return {r["rel"]: ("train_pool" if h(r["rel"]) % 100 < 70 else
                       "gen_dev" if h(r["rel"]) % 100 < 80 else "gen_test") for r in rows}


def main() -> int:
    excl_path = OUT / "excluded.txt"
    excluded = set(excl_path.read_text(encoding="utf-8").split()) if excl_path.is_file() else set()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for corpus in CORPORA:
        rows = list_corpus(corpus)
        a = assign(corpus, rows)
        assert a == assign(corpus, list(reversed(rows))), "split must not depend on listing order"
        groups = {}
        for r in rows:
            groups.setdefault(r["group"], set()).add(a[r["rel"]])
        assert all(len(s) == 1 for s in groups.values()), f"{corpus}: a split unit spans splits"
        for s in SPLITS:
            sel = [r for r in rows if a[r["rel"]] == s and r["rel"] not in excluded]
            (OUT / f"{corpus}_{s}.txt").write_text("".join(f"{r['rel']}\n" for r in sel), encoding="utf-8")
            summary.append({"corpus": corpus, "split": s, "files": len(sel),
                            "units": len({r["group"] for r in sel}),
                            "excluded": sum(1 for r in rows if a[r["rel"]] == s and r["rel"] in excluded),
                            "share_of_corpus": round(sum(1 for r in rows if a[r["rel"]] == s) / len(rows), 3),
                            "units_list": ";".join(sorted({r["group"] for r in sel})) if corpus == "demand" else ""})
        lists = [set((OUT / f"{corpus}_{s}.txt").read_text(encoding="utf-8").split()) for s in SPLITS]
        assert not (lists[0] & lists[1] or lists[0] & lists[2] or lists[1] & lists[2]), corpus
    df = pd.DataFrame(summary)
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
