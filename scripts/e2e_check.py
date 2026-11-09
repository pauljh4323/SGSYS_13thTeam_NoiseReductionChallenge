"""End-to-end check of the submission entry point main.py.

(a) copy all validation noisy wavs to outputs/e2e/input/, run main.py as a subprocess,
    then score the written files with scripts/eval.py --pred-dir.
(b) git clone the local repo (HEAD) to outputs/e2e/clone/ and run the clone's main.py on the
    same input -> outputs/e2e/output_clone/.
(c) run the local main.py with cwd=outputs/ (absolute --input/--output) -> outputs/e2e/output_cwd/;
    relative paths in the config (e.g. weights) must resolve against the repo, not the cwd.
(d) fresh pip venv (outputs/venv_ci, created from this interpreter): pip install -r the clone's
    requirements.txt, run the clone's main.py -> outputs/e2e/output_venv/; venv deleted afterwards.
Asserts all output sets are bit-identical and match the inputs (names/count/length/sr/subtype),
then checks a 44.1 kHz stereo file (in a subfolder, .WAV) and a 1.7 s file.
Challenge 1 compliance: the submission method must be in CLASSICAL_METHODS, and main.py run in a
subprocess must not import torch.
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
import yaml
from scipy.signal import resample_poly

from _common import REPO_ROOT

from nrc.io import load_manifest, resolve_data_root

E2E = REPO_ROOT / "outputs" / "e2e"
VENV = REPO_ROOT / "outputs" / "venv_ci"
# Challenge 1 = traditional DSP only. Add new classical methods here; never a learned model.
CLASSICAL_METHODS = {"noisy", "wiener", "mmse_lsa"}
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


def run_main(main_py: Path, inp: Path, out: Path, cwd: Path | None = None,
             python: Path | str = sys.executable) -> int:
    """Run main.py (cwd defaults to its repo); return its reported failure count."""
    rmtree(out)
    stdout = run([python, main_py, "--input", inp.resolve(), "--output", out.resolve()],
                 cwd=cwd or main_py.parent)
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


def same_bits(a: Path, b: Path) -> tuple[bool, int]:
    diff = [r for r in wavs(a) if not (b / r).is_file() or (a / r).read_bytes() != (b / r).read_bytes()]
    return wavs(a) == wavs(b) and not diff, len(diff)


TORCH_PROBE = """import runpy, sys
main_py, inp, out = sys.argv[1:4]
sys.argv = [main_py, "--input", inp, "--output", out]
try:
    runpy.run_path(main_py, run_name="__main__")
except SystemExit as e:
    if e.code not in (0, None):
        raise
print("TORCH_LOADED=" + str("torch" in sys.modules))
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)
    root = resolve_data_root(args.data_root)
    manifest = load_manifest(root, "validation").sort_values("segment_id")

    sub_cfg = yaml.safe_load((REPO_ROOT / "configs" / "submission.yaml").read_text(encoding="utf-8"))
    check(sub_cfg.get("method") in CLASSICAL_METHODS,
          f"(compliance) submission method {sub_cfg.get('method')!r} in CLASSICAL_METHODS "
          f"{sorted(CLASSICAL_METHODS)}")

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

    # (d) fresh pip venv with only requirements.txt, running the clone's main.py
    out_d = E2E / "output_venv"
    rmtree(VENV)
    try:
        run([sys.executable, "-m", "venv", VENV], cwd=REPO_ROOT)
        py = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([py, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
             "-r", clone / "requirements.txt"], cwd=clone)
        frozen = run([py, "-m", "pip", "freeze"], cwd=clone).split()
        print("venv packages: " + " ".join(frozen))
        check(run_main(clone / "main.py", inp, out_d, python=py) == 0,
              "(d) fresh pip venv: clone main.py failed=0")
        compare_to_inputs(inp, out_d, "(d)")
        ok, ndiff = same_bits(out_a, out_d)
        check(ok, f"(a)==(d) fresh pip venv bit-identical ({len(wavs(out_a))} files, {ndiff} differ)")
    finally:
        rmtree(VENV)
    check(not VENV.exists(), "(d) venv deleted")

    # (c) local main.py launched from another working directory
    out_c = E2E / "output_cwd"
    check(run_main(REPO_ROOT / "main.py", inp, out_c, cwd=REPO_ROOT / "outputs") == 0,
          "(c) main.py from cwd=outputs/ failed=0")
    diff = [r for r in wavs(out_a) if not (out_c / r).is_file()
            or (out_a / r).read_bytes() != (out_c / r).read_bytes()]
    check(wavs(out_a) == wavs(out_c) and not diff,
          f"(a)==(c) bit-identical ({len(wavs(out_a))} files, {len(diff)} differ)")

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

    # compliance: main.py in a subprocess must not import torch
    out_t = E2E / "output_torchfree"
    rmtree(out_t)
    stdout = run([sys.executable, "-c", TORCH_PROBE, REPO_ROOT / "main.py", x_in.resolve(),
                  out_t.resolve()], cwd=REPO_ROOT)
    flag = [ln for ln in stdout.splitlines() if ln.startswith("TORCH_LOADED=")]
    check(flag == ["TORCH_LOADED=False"], f"(compliance) main.py subprocess: 'torch' not in sys.modules "
                                          f"({flag[-1] if flag else 'no probe output'})")

    print(f"\nE2E {'PASSED' if not FAILS else 'FAILED'}: {len(FAILS)} failing check(s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
