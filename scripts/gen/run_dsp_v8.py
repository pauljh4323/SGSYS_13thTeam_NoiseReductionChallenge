"""Enhance a split with dsp-v8 exactly as tagged (exp/r2-gen).

r2-prep's mmse_lsa predates the NLM a priori SNR that dsp-v8 needs, so dsp-v8 runs from a worktree checked out
at tag dsp-v8 (git worktree add --detach data_ext/wt_dsp_v8 dsp-v8) with that tag's configs/submission.yaml.
Writes outputs/dsp_v8_pred/<split>/<noisy filename> (16-bit, as main.py writes); existing files are skipped
(resumable). Score with: python scripts/eval.py --config configs/gen/dsp_v8_pred.yaml --split <split>
--pred-dir outputs/dsp_v8_pred/<split>.
Run: python scripts/gen/run_dsp_v8.py --split <split> [--worktree data_ext/wt_dsp_v8]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--worktree", default=str(REPO / "data_ext" / "wt_dsp_v8"))
    args = ap.parse_args(argv)
    wt = Path(args.worktree).resolve()
    sys.path.insert(0, str(wt / "src"))
    from nrc.io import to_pcm16  # noqa: E402  (worktree code)
    from nrc.methods import get_method  # noqa: E402
    import nrc  # noqa: E402
    assert Path(nrc.__file__).resolve().is_relative_to(wt), nrc.__file__
    cfg = yaml.safe_load((wt / "configs" / "submission.yaml").read_text(encoding="utf-8"))
    method = get_method(cfg["method"], **cfg["params"])
    # manifest location from THIS repo's registry (gen splits are not known to the worktree)
    official = args.split in ("train", "validation")  # official manifest paths are relative to the data root
    split_dir = (REPO / "data" / "noise_reduction_challenge" if official
                 else {"val_remix": REPO / "data" / "val_remix"}.get(args.split, REPO / "data" / "gen" / args.split))
    man = pd.read_csv(split_dir / (f"{args.split}/samples.csv" if official else "samples.csv"), encoding="utf-8-sig")
    out = REPO / "outputs" / "dsp_v8_pred" / args.split
    out.mkdir(parents=True, exist_ok=True)
    t0, done = time.perf_counter(), 0
    for _, r in man.iterrows():
        dst = out / Path(r.noisy_path).name
        if dst.is_file():
            continue
        x, sr = sf.read(str(split_dir / r.noisy_path), dtype="float64")
        y = method(x, sr)
        assert y.shape == x.shape
        pcm, _ = to_pcm16(y)
        sf.write(str(dst), pcm, sr, subtype="PCM_16")
        done += 1
    print(f"dsp_v8 {args.split}: {done} enhanced, {len(man)} total, {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
