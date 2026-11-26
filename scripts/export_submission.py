"""Build the submission repository tree (sig_NalS_ANd_sYS_tem-team13-NoiseRC) from a tag of this repo.

    python scripts/export_submission.py --tag r2-ready --target ../sig_NalS_ANd_sYS_tem-team13-NoiseRC

Every exported file is read from the git tag (never the working tree):
  main.py, the src/ modules main.py actually uses with configs/submission.yaml (traced 2026-11-26 with an
  "open" audit hook + sys.modules on 5 validation files, and a static import scan), requirements.txt
  (unchanged), weights/<submitted>.pt only, plus generated files: a self-contained configs/submission.yaml
  (method/params inlined, no configs/dl/ reference), .gitignore, input/.gitkeep, README.md
  (docs/submission/README_template.md).
One change to the exported code: src/nrc/methods/__init__.py imports and registers only gru_mask, so the
classical DSP methods (mmse_lsa, noisy, wiener, stft) are not shipped. The gru_mask path is unchanged.
The target must be empty, missing, or a git checkout; everything in it except .git/ is deleted first.
Writes docs/submission/MANIFEST.txt in this repo (source tag/commit, size + SHA-256 per exported file).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "docs" / "submission" / "README_template.md"
MANIFEST = REPO_ROOT / "docs" / "submission" / "MANIFEST.txt"
WEIGHTS = "weights/gru_v8p2z3e.pt"
MODEL_CONFIG = "configs/dl/gru_v8p2z3e.yaml"   # must match configs/submission.yaml method/params
MAX_WEIGHT_BYTES = 50 * 2 ** 20
COPIED = [                                     # verbatim from the tag
    "main.py",
    "requirements.txt",
    "src/nrc/__init__.py",
    "src/nrc/io.py",
    "src/nrc/methods/gru_mask.py",
    "src/nrc/models/__init__.py",
    "src/nrc/models/gru_mask.py",
    WEIGHTS,
]
METHODS_INIT = "src/nrc/methods/__init__.py"   # copied with PATCHES applied
PATCHES = [
    ("from . import gru_mask, mmse_lsa, noisy, wiener\n", "from . import gru_mask\n"),
    ('    "gru_mask": gru_mask.enhance,  # round 2 (DL); torch is imported lazily on first call\n'
     '    "mmse_lsa": mmse_lsa.enhance,\n'
     '    "noisy": noisy.enhance,\n'
     '    "wiener": wiener.enhance,\n',
     '    "gru_mask": gru_mask.enhance,  # round 2 (DL); torch is imported lazily on first call\n'),
]
CONFIG_HEADER = (
    "# Submission config (round 2): gru_v8p2z3e + input level mode rms_norm (inference only).\n"
    "# rms_norm: the input is scaled to target_rms (median whole-file RMS of the train noisy files) before\n"
    "# the model; the output stays at that level (peak capped at 0.99). weights is relative to the repo root.\n"
)
CONFIG_BODY = (
    "method: gru_mask\n"
    "params:\n"
    "  weights: {weights}\n"
    "  device: {device}\n"
    "  level_norm: {level_norm}\n"
    "  level_mode: {level_mode}\n"
    "  target_rms: {target_rms:.6f}\n"
)
GITIGNORE = "output/\n__pycache__/\n.venv/\n"
ABS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/home/|/Users/|/content/")  # = scripts/e2e_check.py


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True).stdout


def blob(tag: str, path: str) -> bytes:
    return git("cat-file", "blob", f"{tag}:{path}")


def rm_readonly(func, p, _exc):  # git objects are read-only on Windows
    Path(p).chmod(stat.S_IWRITE)
    func(p)


def prepare_target(target: Path) -> None:
    if target == REPO_ROOT or target.is_relative_to(REPO_ROOT) or REPO_ROOT.is_relative_to(target):
        sys.exit(f"error: target {target} overlaps the source repo")
    if target.exists():
        entries = list(target.iterdir())
        if entries and not (target / ".git").is_dir():
            sys.exit(f"error: target {target} is not empty and not a git checkout")
        for p in entries:
            if p.name == ".git":
                continue
            shutil.rmtree(p, onerror=rm_readonly) if p.is_dir() else p.unlink()
    target.mkdir(parents=True, exist_ok=True)


def build(tag: str) -> dict[str, bytes]:
    files = {path: blob(tag, path) for path in COPIED}

    text = blob(tag, METHODS_INIT).decode("utf-8")
    for old, new in PATCHES:
        if text.count(old) != 1:
            sys.exit(f"error: patch for {METHODS_INIT} does not apply at {tag}: {old!r}")
        text = text.replace(old, new)
    files[METHODS_INIT] = text.encode("utf-8")

    sub = yaml.safe_load(blob(tag, "configs/submission.yaml"))
    model = yaml.safe_load(blob(tag, MODEL_CONFIG))
    if (sub["method"], sub["params"]) != (model["method"], model["params"]) or sub["params"]["weights"] != WEIGHTS:
        sys.exit(f"error: configs/submission.yaml at {tag} is not {MODEL_CONFIG} / {WEIGHTS}")
    p = sub["params"]
    if set(p) != {"weights", "device", "level_norm", "level_mode", "target_rms"}:
        sys.exit(f"error: unexpected submission params {sorted(p)}")
    cfg = CONFIG_HEADER + CONFIG_BODY.format(weights=p["weights"], device=p["device"],
                                             level_norm=str(p["level_norm"]).lower(),
                                             level_mode=p["level_mode"], target_rms=p["target_rms"])
    if yaml.safe_load(cfg) != {"method": sub["method"], "params": p}:
        sys.exit("error: generated configs/submission.yaml does not round-trip")
    files["configs/submission.yaml"] = cfg.encode("utf-8")

    files[".gitignore"] = GITIGNORE.encode("utf-8")
    files["input/.gitkeep"] = b""
    readme = TEMPLATE.read_text(encoding="utf-8")
    for needed in ("pip install -r requirements.txt", "python main.py", WEIGHTS, "configs/submission.yaml"):
        if needed not in readme:
            sys.exit(f"error: README template lacks {needed!r}")
    files["README.md"] = readme.encode("utf-8")
    return files


def self_check(files: dict[str, bytes]) -> None:
    allowed_prefixes = ("src/nrc/", "configs/submission.yaml", "weights/", "input/.gitkeep")
    top = {"main.py", "requirements.txt", ".gitignore", "README.md"}
    for path, data in files.items():
        if not (path in top or path.startswith(allowed_prefixes)):
            sys.exit(f"error: unexpected exported path {path}")
        if path.startswith("weights/") and path != WEIGHTS:
            sys.exit(f"error: extra weights {path}")
        if path.endswith((".py", ".yaml", ".txt")):
            for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
                if ABS_PATH.search(re.sub(r"\w+://\S+", "", line)):
                    sys.exit(f"error: absolute path in {path}:{n}")
    w = files[WEIGHTS]
    if not 2 ** 20 < len(w) < MAX_WEIGHT_BYTES or w.startswith(b"version https://git-lfs"):
        sys.exit(f"error: {WEIGHTS} is {len(w)} B (need 1 MB < size < 50 MB, no LFS pointer)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="source tag on r2-prep, e.g. r2-ready")
    ap.add_argument("--target", required=True, help="submission repo checkout (outside this repo)")
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args(argv)

    commit = git("rev-parse", f"{args.tag}^{{commit}}").decode().strip()
    files = build(args.tag)
    self_check(files)
    target = Path(args.target).resolve()
    prepare_target(target)
    for path, data in sorted(files.items()):
        dst = target / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    lines = [
        "# Submission repo export manifest (scripts/export_submission.py; not exported)",
        "repo: https://github.com/pauljh4323/sig_NalS_ANd_sYS_tem-team13-NoiseRC",
        f"source_tag: {args.tag}",
        f"source_commit: {commit}",
        f"files: {len(files)}",
        f"total_bytes: {sum(map(len, files.values()))}",
        "",
        "sha256  bytes  path",
    ]
    lines += [f"{hashlib.sha256(d).hexdigest()}  {len(d)}  {p}" for p, d in sorted(files.items())]
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"exported {len(files)} files from {args.tag} ({commit[:7]}) -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
