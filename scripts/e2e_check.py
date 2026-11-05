"""End-to-end check of the submission entry point main.py.

(a) copy all validation noisy wavs to outputs/e2e/input/, run main.py as a subprocess,
    then score the written files with scripts/eval.py --pred-dir.
(b) git clone the local repo (HEAD) to outputs/e2e/clone/ and run the clone's main.py on the
    same input -> outputs/e2e/output_clone/.
Asserts both output sets are bit-identical and match the inputs (names/count/length/sr/subtype),
then checks a 44.1 kHz stereo file (in a subfolder, .WAV) and a 1.7 s file.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from _common import REPO_ROOT

from nrc.io import load_manifest, resolve_data_root

E2E = REPO_ROOT / "outputs" / "e2e"
FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def rmtree(path: Path) -> None:
    def onerror(func, p, _exc):  # git objects are read-only on Windows
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.exists():
        shutil.rmtree(path, onerror=onerror)


def run(cmd, cwd) -> str:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.stderr.strip():
        print(r.stderr.rstrip())
    if r.returncode != 0:
        print(r.stdout)
        raise RuntimeError(f"command failed ({r.returncode})")
    return r.stdout


def run_main(main_py: Path, inp: Path, out: Path) -> int:
    """Run main.py; return its reported failure count."""
    rmtree(out)
    stdout = run([sys.executable, main_py, "--input", inp, "--output", out], cwd=main_py.parent)
    summary = stdout.strip().splitlines()[-1]
    print(summary)
    m = re.search(r"failed=(\d+)", summary)
    return int(m.group(1)) if m else -1


def wavs(d: Path) -> list[Path]:
    return sorted(p.relative_to(d) for p in d.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")


def compare_to_inputs(inp: Path, out: Path, label: str, require_pcm16: bool = True) -> None:
    rels_in, rels_out = wavs(inp), wavs(out)
    check(rels_in == rels_out, f"{label}: same filenames/count ({len(rels_out)}/{len(rels_in)})")
    bad = []
    for rel in rels_in:
        if not (out / rel).is_file():
            continue
        a, b = sf.info(str(inp / rel)), sf.info(str(out / rel))
        if (a.samplerate, a.frames, a.channels) != (b.samplerate, b.frames, b.channels) \
                or (require_pcm16 and b.subtype != "PCM_16") or a.subtype != b.subtype:
            bad.append(f"{rel}: in {a.samplerate}/{a.frames}/{a.channels}/{a.subtype} "
                       f"out {b.samplerate}/{b.frames}/{b.channels}/{b.subtype}")
    check(not bad, f"{label}: sr/length/channels/subtype preserved" + (f" {bad[:3]}" if bad else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)
    root = resolve_data_root(args.data_root)
    manifest = load_manifest(root, "validation").sort_values("segment_id")

    # (a) local repo
    inp, out_a = E2E / "input", E2E / "output"
    rmtree(inp)
    inp.mkdir(parents=True)
    for rel in manifest.noisy_path:
        shutil.copyfile(root / rel, inp / Path(rel).name)
    check(len(wavs(inp)) == len(manifest), f"(a) copied {len(manifest)} validation noisy wavs")
    check(run_main(REPO_ROOT / "main.py", inp, out_a) == 0, "(a) main.py failed=0")
    compare_to_inputs(inp, out_a, "(a)")
    if not args.skip_eval:
        stdout = run([sys.executable, REPO_ROOT / "scripts" / "eval.py",
                      "--config", REPO_ROOT / "configs" / "submission.yaml",
                      "--split", "validation", "--pred-dir", out_a, "--jobs", args.jobs,
                      "--notes", "e2e main.py outputs"], cwd=REPO_ROOT)
        print("\n".join(stdout.splitlines()[:6]))

    # (b) fresh clone of HEAD
    clone, out_b = E2E / "clone", E2E / "output_clone"
    rmtree(clone)
    run(["git", "clone", "--quiet", "--no-hardlinks", REPO_ROOT, clone], cwd=REPO_ROOT)
    head = run(["git", "rev-parse", "--short", "HEAD"], cwd=clone).strip()
    print(f"clone HEAD {head}")
    check(not (clone / "data").exists(), "(b) clone has no data/")
    check(run_main(clone / "main.py", inp, out_b) == 0, "(b) clone main.py failed=0")
    compare_to_inputs(inp, out_b, "(b)")
    diff = [r for r in wavs(out_a) if not (out_b / r).is_file()
            or (out_a / r).read_bytes() != (out_b / r).read_bytes()]
    check(wavs(out_a) == wavs(out_b) and not diff,
          f"(a)==(b) bit-identical ({len(wavs(out_a))} files, {len(diff)} differ)")

    # extra: 44.1 kHz stereo in a subfolder with .WAV, and a 1.7 s file
    x_in, x_out = E2E / "extra_input", E2E / "extra_output"
    rmtree(x_in)
    (x_in / "sub").mkdir(parents=True)
    src = inp / Path(manifest.noisy_path.iloc[0]).name
    x, sr = sf.read(str(src), dtype="float64")
    x44 = resample_poly(x, 441, 160)
    stereo = np.stack([x44, 0.8 * np.roll(x44, 7)], axis=1)
    sf.write(str(x_in / "sub" / "stereo44k.WAV"), np.clip(stereo, -1, 0.9999), 44100,
             subtype="PCM_16")
    sf.write(str(x_in / "short_1p7s.wav"), x[: int(1.7 * sr)], sr, subtype="PCM_16")
    check(run_main(REPO_ROOT / "main.py", x_in, x_out) == 0, "(extra) main.py failed=0")
    compare_to_inputs(x_in, x_out, "(extra)")
    for rel in wavs(x_in):
        info = sf.info(str(x_in / rel))
        same = (x_in / rel).read_bytes() == (x_out / rel).read_bytes()
        check(not same, f"(extra) {rel.as_posix()} enhanced (not passthrough): "
                        f"{info.samplerate} Hz, {info.channels} ch, {info.frames} frames")

    print(f"\nE2E {'PASSED' if not FAILS else 'FAILED'}: {len(FAILS)} failing check(s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
