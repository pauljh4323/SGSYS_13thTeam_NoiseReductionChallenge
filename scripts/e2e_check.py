"""End-to-end check of the submission entry point main.py.

(a) copy all validation noisy wavs to outputs/e2e/input/, run main.py as a subprocess,
    then score the written files with scripts/eval.py --pred-dir.
(b) git clone the local repo (HEAD, or --ref) to outputs/e2e/clone/ and run the clone's main.py on the
    same input -> outputs/e2e/output_clone/.
(c) run the local main.py with cwd=outputs/ (absolute --input/--output) -> outputs/e2e/output_cwd/;
    relative paths in the config (e.g. weights) must resolve against the repo, not the cwd.
(d) fresh pip venv (outputs/venv_ci, created from this interpreter): pip install -r the clone's
    requirements.txt, then the README command as written: validation inputs copied to the clone's
    input/, `python main.py` (no arguments, cwd = clone) -> clone/output/; venv deleted afterwards.
Asserts all output sets are bit-identical and match the inputs (names/count/length/sr/subtype),
that each output dir holds exactly the input WAV names and nothing else (no logs/extra files),
then checks a 44.1 kHz stereo file (in a subfolder, .WAV) and a 1.7 s file.
Compliance depends on ROUND (override with --round):
  ROUND 1: the submission method must be in CLASSICAL_METHODS, and main.py run in a subprocess must
           not import torch.
  ROUND 2: the submission method must be in CLASSICAL_METHODS | ROUND2_METHODS (torch allowed).
(d) also reports the fresh-venv pip install time and venv size.
--ref <tag|branch|commit> clones that ref instead of HEAD (use the submission tag before submitting).
--repo-url <url> clones that repository (e.g. the submission repo built by scripts/export_submission.py)
instead of this one for (b) and (d); --ref then names a ref of that repo (default: its HEAD = ls-remote
HEAD), the compliance method check also reads the clone's configs/submission.yaml, and (a)==(b) /
(a)==(d) compare its outputs bit for bit with this checkout's main.py.
Ends with the TA notice checklist (CHECKLIST, 8 items), each PASS only if all its checks passed.
Item 8 only checks that pesq 0.0.4 returns a finite score per file (in a subprocess, so a native
crash is a FAIL); the metric itself comes from scripts/eval.py in (a).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
import time
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
# Challenge 2 = deep learning; learned methods allowed only when ROUND == 2.
ROUND2_METHODS = {"gru_mask"}
ROUND = 2  # main: 1 (classical only + torch-free check); r2-prep: 2
FAILS: list[str] = []
# TA round-2 notice: pre-submission checklist (printed at the end).
CHECKLIST = {
    1: "fresh clone of the tag/ref runs",
    2: "fresh venv install from requirements.txt",
    3: "README install + inference commands run",
    4: "weights present (no Git LFS pointer files)",
    5: "no absolute paths (main.py, src/, configs/, requirements.txt)",
    6: "exactly one output per input, directly under output/",
    7: "outputs 16 kHz mono 48000 samples",
    8: "PESQ computable on every validation output",
}
ITEMS: dict[int, list[bool]] = {k: [] for k in CHECKLIST}
VAL_FORMAT = (16000, 1, 48000)  # hidden test and validation: 16 kHz mono, 3 s
ABS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/home/|/Users/|/content/")
LFS_HEAD = b"version https://git-lfs"


def check(ok: bool, msg: str, item: int | None = None) -> None:
    tag = f"[{item}] " if item else ""
    print(("PASS " if ok else "FAIL ") + tag + msg, flush=True)
    if item:
        ITEMS[item].append(ok)
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


def main_failed(stdout: str) -> int:
    summary = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    print(summary)
    m = re.search(r"failed=(\d+)", summary)
    return int(m.group(1)) if m else -1


def run_main(main_py: Path, inp: Path, out: Path, cwd: Path | None = None,
             python: Path | str = sys.executable) -> int:
    """Run main.py (cwd defaults to its repo); return its reported failure count."""
    rmtree(out)
    return main_failed(run([python, main_py, "--input", inp.resolve(), "--output", out.resolve()],
                           cwd=cwd or main_py.parent))


def wavs(d: Path) -> list[Path]:
    return sorted(p.relative_to(d) for p in d.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")


def compare_to_inputs(inp: Path, out: Path, label: str, require_pcm16: bool = True,
                      items: tuple[int | None, int | None] = (None, None)) -> None:
    rels_in, rels_out = wavs(inp), wavs(out)
    check(rels_in == rels_out, f"{label}: same filenames/count ({len(rels_out)}/{len(rels_in)})",
          items[0])
    bad = []
    for rel in rels_in:
        if not (out / rel).is_file():
            continue
        a, b = sf.info(str(inp / rel)), sf.info(str(out / rel))
        if (a.samplerate, a.frames, a.channels) != (b.samplerate, b.frames, b.channels) \
                or (require_pcm16 and b.subtype != "PCM_16") or a.subtype != b.subtype:
            bad.append(f"{rel}: in {a.samplerate}/{a.frames}/{a.channels}/{a.subtype} "
                       f"out {b.samplerate}/{b.frames}/{b.channels}/{b.subtype}")
    check(not bad, f"{label}: sr/length/channels/subtype preserved" + (f" {bad[:3]}" if bad else ""),
          items[1])


def exact_contents(inp: Path, out: Path, label: str, item: int | None = None) -> None:
    """output/ must hold exactly the input WAV paths (same names, same layout) and nothing else:
    no extra files of any type, no extra or missing directories."""
    want = {p.as_posix() for p in wavs(inp)}
    want |= {parent.as_posix() for p in wavs(inp) for parent in p.parents if parent != Path(".")}
    got = {p.relative_to(out).as_posix() for p in out.rglob("*")}
    extra, missing = sorted(got - want), sorted(want - got)
    check(not extra and not missing,
          f"{label}: output/ holds exactly the input WAV names ({len(got)} entries)"
          + (f" extra {extra[:3]}" if extra else "") + (f" missing {missing[:3]}" if missing else ""),
          item)


def flat_val_outputs(inp: Path, out: Path, label: str) -> None:
    """Validation run: one WAV per input directly under output/ (no subfolders), and every output
    16 kHz mono 48000 samples like the hidden test (items 6, 7)."""
    entries = list(out.iterdir()) if out.is_dir() else []
    nested = [p.name for p in entries if not p.is_file()]
    check(not nested and len(wavs(out)) == len(wavs(inp)) and all(len(r.parts) == 1 for r in wavs(out)),
          f"{label}: {len(wavs(out))} outputs for {len(wavs(inp))} inputs, all directly under output/"
          + (f" (subfolders {nested[:3]})" if nested else ""), 6)
    bad = []
    for rel in wavs(out):
        i = sf.info(str(out / rel))
        if (i.samplerate, i.channels, i.frames) != VAL_FORMAT:
            bad.append(f"{rel}: {i.samplerate} Hz / {i.channels} ch / {i.frames}")
    check(not bad and bool(wavs(out)), f"{label}: every output {VAL_FORMAT[0]} Hz, mono, "
                                       f"{VAL_FORMAT[2]} samples" + (f" {bad[:3]}" if bad else ""), 7)


def same_bits(a: Path, b: Path) -> tuple[bool, int]:
    diff = [r for r in wavs(a) if not (b / r).is_file() or (a / r).read_bytes() != (b / r).read_bytes()]
    return wavs(a) == wavs(b) and not diff, len(diff)


def weight_problems(root: Path) -> tuple[list[str], list[str]]:
    """Weights named by configs/submission.yaml and configs/dl/*.yaml: present, > 1 MB, not an LFS pointer."""
    names = set()
    for cfg in [root / "configs" / "submission.yaml", *sorted((root / "configs" / "dl").glob("*.yaml"))]:
        params = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("params") or {}
        if params.get("weights"):
            names.add(params["weights"])
    bad = []
    for name in sorted(names):
        p = root / name
        if not p.is_file():
            bad.append(f"{name} missing")
        elif p.stat().st_size < 2 ** 20 or p.read_bytes()[:len(LFS_HEAD)] == LFS_HEAD:
            bad.append(f"{name} is an LFS pointer or too small ({p.stat().st_size} B)")
    return sorted(names), bad


def abs_path_hits(root: Path) -> list[str]:
    """Absolute paths (drive letters, /home/, /Users/, /content/) in the inference files; URLs ignored."""
    files = [root / "main.py", root / "requirements.txt",
             *sorted(root.glob("src/**/*.py")), *sorted(root.glob("configs/**/*.yaml"))]
    hits = []
    for f in files:
        if not f.is_file():
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if ABS_PATH.search(re.sub(r"\w+://\S+", "", line)):
                hits.append(f"{f.relative_to(root).as_posix()}:{n}")
    return hits


PESQ_PROBE = """import math, sys
import soundfile as sf
from pesq import pesq
ok = bad = 0
for line in open(sys.argv[1], encoding="utf-8"):
    clean, enh = line.rstrip("\\n").split("\\t")
    try:
        c, _ = sf.read(clean, dtype="float64")
        e, _ = sf.read(enh, dtype="float64")
        s = pesq(16000, c, e, "wb")
        good = math.isfinite(s)
    except Exception as exc:  # noqa: BLE001
        good = False
        print("PESQ_ERR", enh, type(exc).__name__, exc)
    ok += good
    bad += not good
print(f"PESQ_OK={ok} PESQ_BAD={bad}")
"""


def pesq_computable(root: Path, manifest, out: Path, label: str) -> None:
    """Item 8: pesq 0.0.4 WB gives a finite score for every validation output (clean = reference,
    evaluation only). Runs in a subprocess so a native pesq crash is a FAIL, not an abort."""
    pairs = E2E / "pesq_pairs.tsv"
    pairs.write_text("".join(f"{(root / c).resolve()}\t{(out / Path(n).name).resolve()}\n"
                             for c, n in zip(manifest.clean_path, manifest.noisy_path)), encoding="utf-8")
    r = subprocess.run([sys.executable, "-c", PESQ_PROBE, pairs], capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.startswith("PESQ_ERR"):
            print(ln)
    m = re.search(r"PESQ_OK=(\d+) PESQ_BAD=(\d+)", r.stdout)
    n_ok = int(m.group(1)) if m else 0
    check(r.returncode == 0 and m is not None and n_ok == len(manifest) and m.group(2) == "0",
          f"{label}: pesq 0.0.4 WB finite on {n_ok}/{len(manifest)} outputs (exit code {r.returncode})", 8)


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
    parser.add_argument("--round", type=int, choices=(1, 2), default=ROUND,
                        help=f"challenge round for the compliance checks (default {ROUND})")
    parser.add_argument("--ref", default="HEAD",
                        help="tag/branch/commit to clone for (b) and (d), e.g. the submission tag")
    parser.add_argument("--repo-url", default=None,
                        help="clone this repository for (b) and (d) instead of the local repo "
                             "(e.g. the submission repo); (a)/(c) still run the local main.py")
    args = parser.parse_args(argv)
    root = resolve_data_root(args.data_root)
    manifest = load_manifest(root, "validation").sort_values("segment_id")

    allowed = CLASSICAL_METHODS if args.round == 1 else CLASSICAL_METHODS | ROUND2_METHODS
    print(f"ROUND {args.round}: allowed methods {sorted(allowed)}", flush=True)
    sub_cfg = yaml.safe_load((REPO_ROOT / "configs" / "submission.yaml").read_text(encoding="utf-8"))
    check(sub_cfg.get("method") in allowed,
          f"(compliance) round {args.round}: submission method {sub_cfg.get('method')!r} in "
          f"{sorted(allowed)}")

    # (a) local repo
    inp, out_a = E2E / "input", E2E / "output"
    rmtree(inp)
    inp.mkdir(parents=True)
    for rel in manifest.noisy_path:
        shutil.copyfile(root / rel, inp / Path(rel).name)
    check(len(wavs(inp)) == len(manifest), f"(a) copied {len(manifest)} validation noisy wavs")
    check(run_main(REPO_ROOT / "main.py", inp, out_a) == 0, "(a) main.py failed=0")
    compare_to_inputs(inp, out_a, "(a)", items=(6, 7))
    exact_contents(inp, out_a, "(a)", 6)
    flat_val_outputs(inp, out_a, "(a)")
    if not args.skip_eval:
        stdout = run([sys.executable, REPO_ROOT / "scripts" / "eval.py",
                      "--config", REPO_ROOT / "configs" / "submission.yaml",
                      "--split", "validation", "--pred-dir", out_a, "--jobs", args.jobs,
                      "--notes", "e2e main.py outputs"], cwd=REPO_ROOT)
        print("\n".join(stdout.splitlines()[:6]))

    # (b) fresh clone of --ref (default HEAD)
    clone, out_b = E2E / "clone", E2E / "output_clone"
    rmtree(clone)
    if args.repo_url:
        run(["git", "clone", "--quiet", args.repo_url, clone], cwd=REPO_ROOT)
        if args.ref == "HEAD":
            want = run(["git", "ls-remote", args.repo_url, "HEAD"], cwd=REPO_ROOT).split()[0]
        else:
            ref = args.ref
            if subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=clone,
                              capture_output=True).returncode != 0:
                ref = f"origin/{ref}"  # a branch of the cloned repo
            want = run(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=clone).strip()
        print(f"repo: {args.repo_url}")
    else:
        want = run(["git", "rev-parse", f"{args.ref}^{{commit}}"], cwd=REPO_ROOT).strip()
        run(["git", "clone", "--quiet", "--no-hardlinks", REPO_ROOT, clone], cwd=REPO_ROOT)
    if args.ref != "HEAD":
        run(["git", "-c", "advice.detachedHead=false", "checkout", "--quiet", want], cwd=clone)
    head = run(["git", "rev-parse", "HEAD"], cwd=clone).strip()
    print(f"clone of {args.ref}: {head[:7]}")
    check(head == want, f"(b) clone checked out {args.ref} = {want[:7]}", 1)
    check(not (clone / "data").exists(), "(b) clone has no data/", 1)
    if args.repo_url:
        clone_cfg = yaml.safe_load((clone / "configs" / "submission.yaml").read_text(encoding="utf-8"))
        check(clone_cfg.get("method") in allowed,
              f"(compliance) clone submission method {clone_cfg.get('method')!r} in {sorted(allowed)}")
    names, bad_w = weight_problems(clone)
    check(bool(names) and not bad_w, f"(b) clone weights present, no LFS pointers ({len(names)} files"
          + (f"; {bad_w[:3]})" if bad_w else ")"), 4)
    hits = abs_path_hits(clone)
    check(not hits, "(b) no absolute paths in main.py, src/, configs/, requirements.txt"
          + (f" {hits[:5]}" if hits else ""), 5)
    check(run_main(clone / "main.py", inp, out_b) == 0, "(b) clone main.py failed=0", 1)
    compare_to_inputs(inp, out_b, "(b)", items=(6, 7))
    exact_contents(inp, out_b, "(b)", 6)
    flat_val_outputs(inp, out_b, "(b)")
    ok, ndiff = same_bits(out_a, out_b)
    check(ok, f"(a)==(b) bit-identical ({len(wavs(out_a))} files, {ndiff} differ)", 1)

    # (d) fresh pip venv with only requirements.txt, then the README command in the clone:
    #     inputs in input/, `python main.py` with no arguments -> output/
    out_d = clone / "output"
    pesq_dir = out_a
    rmtree(VENV)
    try:
        run([sys.executable, "-m", "venv", VENV], cwd=REPO_ROOT)
        py = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        t0 = time.perf_counter()
        try:
            run([py, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                 "-r", clone / "requirements.txt"], cwd=clone)
            pip_ok = True
        except RuntimeError:
            pip_ok = False
        check(pip_ok, "(d) fresh venv: pip install -r requirements.txt", 2)
        if pip_ok:
            size_mb = sum(p.stat().st_size for p in VENV.rglob("*") if p.is_file()) / 2 ** 20
            print(f"venv: pip install {time.perf_counter() - t0:.0f} s, size {size_mb:.0f} MB", flush=True)
            frozen = run([py, "-m", "pip", "freeze"], cwd=clone).split()
            print("venv packages: " + " ".join(frozen))
            # training-only differentiable PESQ (torch-pesq, its torchaudio import, nrc.tpesq) must stay off the inference path
            probe = subprocess.run([py, "-c", "import importlib.util as u; "
                                    "print(*[m for m in ('torch_pesq', 'torchaudio') if u.find_spec(m)])"],
                                   cwd=clone, capture_output=True, text=True)
            check(probe.returncode == 0 and probe.stdout.strip() == "",
                  f"(d) torch_pesq / torchaudio not importable in the venv (found: {probe.stdout.strip() or 'none'})", 2)
            check(not (clone / "src" / "nrc" / "tpesq.py").exists(), "(d) src/nrc/tpesq.py not in the clone", 2)
            readme = (clone / "README.md").read_text(encoding="utf-8")
            check("pip install -r requirements.txt" in readme and "python main.py" in readme,
                  "(d) README states `pip install -r requirements.txt` and `python main.py`", 3)
            (clone / "input").mkdir(exist_ok=True)
            for rel in wavs(inp):
                shutil.copyfile(inp / rel, clone / "input" / rel)
            rmtree(out_d)
            failed = main_failed(run([py, "main.py"], cwd=clone))
            check(failed == 0, "(d) README command `python main.py` (clone, fresh venv, input/ -> output/) failed=0", 3)
            compare_to_inputs(inp, out_d, "(d)", items=(6, 7))
            exact_contents(inp, out_d, "(d)", 6)
            flat_val_outputs(inp, out_d, "(d)")
            ok, ndiff = same_bits(out_a, out_d)
            check(ok, f"(a)==(d) fresh pip venv bit-identical ({len(wavs(out_a))} files, {ndiff} differ)", 3)
            pesq_dir = out_d
    finally:
        rmtree(VENV)
    check(not VENV.exists(), "(d) venv deleted")
    pesq_computable(root, manifest, pesq_dir, "(d)" if pesq_dir == out_d else "(a)")

    # (c) local main.py launched from another working directory
    out_c = E2E / "output_cwd"
    check(run_main(REPO_ROOT / "main.py", inp, out_c, cwd=REPO_ROOT / "outputs") == 0,
          "(c) main.py from cwd=outputs/ failed=0")
    diff = [r for r in wavs(out_a) if not (out_c / r).is_file()
            or (out_a / r).read_bytes() != (out_c / r).read_bytes()]
    check(wavs(out_a) == wavs(out_c) and not diff,
          f"(a)==(c) bit-identical ({len(wavs(out_a))} files, {len(diff)} differ)")
    exact_contents(inp, out_c, "(c)", 6)

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
    exact_contents(x_in, x_out, "(extra)")
    for rel in wavs(x_in):
        info = sf.info(str(x_in / rel))
        same = (x_in / rel).read_bytes() == (x_out / rel).read_bytes()
        check(not same, f"(extra) {rel.as_posix()} enhanced (not passthrough): "
                        f"{info.samplerate} Hz, {info.channels} ch, {info.frames} frames")

    # compliance (round 1 only): main.py in a subprocess must not import torch
    if args.round == 1:
        out_t = E2E / "output_torchfree"
        rmtree(out_t)
        stdout = run([sys.executable, "-c", TORCH_PROBE, REPO_ROOT / "main.py", x_in.resolve(),
                      out_t.resolve()], cwd=REPO_ROOT)
        flag = [ln for ln in stdout.splitlines() if ln.startswith("TORCH_LOADED=")]
        check(flag == ["TORCH_LOADED=False"], f"(compliance) main.py subprocess: 'torch' not in "
                                              f"sys.modules ({flag[-1] if flag else 'no probe output'})")
    else:
        print("SKIP (compliance) torch-free probe: round 2 allows torch", flush=True)

    print(f"\nTA notice checklist ({args.repo_url or 'local repo'}, clone of {args.ref} = {head[:7]}):")
    for k, label in CHECKLIST.items():
        res = ITEMS[k]
        verdict = "PASS" if res and all(res) else "FAIL"
        print(f"  {k}. {verdict}  {label} ({sum(res)}/{len(res)} checks)")
        if not res:
            FAILS.append(f"checklist item {k} not run")
    print(f"\nE2E {'PASSED' if not FAILS else 'FAILED'}: {len(FAILS)} failing check(s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
