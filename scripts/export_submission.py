"""Build the round-2 submission repository tree from a tag on r2-prep.

    python scripts/export_submission.py --tag r2-ready --target ../sig_NalS_ANd_sYS_tem-team13-NoiseRC

Everything is read from the tag with `git cat-file` (never from the working tree), so the export is
reproducible from the tag alone. The target folder (a clone of the submission repo, outside this repo)
is emptied except for .git/ and filled with exactly:
  main.py, src/ (only the modules main.py imports: traced with an "open" audit hook + sys.modules on
  validation files), configs/submission.yaml (self-contained: header + the tag's method/params, no
  reference to configs/dl/), the one weights file named by that config, requirements.txt, .gitignore
  (output/), input/.gitkeep, README.md (from scripts/submission_README.tmpl.md).
The only code change: src/nrc/methods/__init__.py registers gru_mask alone, so the DSP methods
(mmse_lsa, noisy, wiener) and nrc.stft are neither imported nor shipped.
A MANIFEST (SHA-256 + size per file, source tag/commit, exporter commit) is written to
docs/submission_manifest.txt in this repo; it is not exported.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "scripts" / "submission_README.tmpl.md"
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "submission_manifest.txt"

# Traced at r2-ready (main.py + configs/submission.yaml on 5 validation files): repo files opened
# and repo modules in sys.modules, minus the DSP modules that only the registry pulled in.
SOURCE_FILES = (
    "main.py",
    "requirements.txt",
    "src/nrc/__init__.py",
    "src/nrc/io.py",
    "src/nrc/methods/__init__.py",
    "src/nrc/methods/gru_mask.py",
    "src/nrc/models/__init__.py",
    "src/nrc/models/gru_mask.py",
)
# (old, new) replacements in src/nrc/methods/__init__.py; each old must occur exactly once.
REGISTRY_PATCH = (
    ("from . import gru_mask, mmse_lsa, noisy, wiener\n", "from . import gru_mask\n"),
    ('    "mmse_lsa": mmse_lsa.enhance,\n', ""),
    ('    "noisy": noisy.enhance,\n', ""),
    ('    "wiener": wiener.enhance,\n', ""),
)
CONFIG_HEADER = """\
# Submission config (round 2): {model} + input level mode {level_mode} (inference only).
# rms_norm: the input is scaled to target_rms (median whole-file RMS of the train noisy files) before
# the model; the output stays at that level (peak capped at 0.99). weights is relative to the repo root.
"""
GITIGNORE = "output/\n__pycache__/\n.venv/\n"


def git(*args: str) -> bytes:
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def blob(commit: str, path: str) -> bytes:
    """Raw blob bytes at commit:path (no working tree, no line-ending conversion)."""
    return git("cat-file", "blob", f"{commit}:{path}")


def patch_registry(text: str) -> str:
    for old, new in REGISTRY_PATCH:
        if text.count(old) != 1:
            raise SystemExit(f"methods/__init__.py: expected exactly one {old.strip()!r}")
        text = text.replace(old, new)
    # module references only ("noisy" is also the waveform parameter name, which stays)
    for name in ("mmse_lsa", "noisy.enhance", '"noisy":', "wiener"):
        if name in text:
            raise SystemExit(f"methods/__init__.py still mentions {name!r} after the patch")
    return text


def submission_config(commit: str) -> tuple[bytes, str]:
    """Self-contained configs/submission.yaml (header + the tag's method/params block); returns
    (bytes, weights path). Checks the block equals configs/dl/<weights stem>.yaml method/params."""
    text = blob(commit, "configs/submission.yaml").decode("utf-8")
    cfg = yaml.safe_load(text)
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("method:"))
    body = "".join(lines[start:])
    if yaml.safe_load(body) != {"method": cfg["method"], "params": cfg["params"]}:
        raise SystemExit("configs/submission.yaml: the method/params block is not the whole config")
    if "configs/dl" in body:
        raise SystemExit("configs/submission.yaml: method/params block refers to configs/dl/")
    weights = cfg["params"]["weights"]
    stem = Path(weights).stem
    dl = yaml.safe_load(blob(commit, f"configs/dl/{stem}.yaml").decode("utf-8"))
    if (dl["method"], dl["params"]) != (cfg["method"], cfg["params"]):
        raise SystemExit(f"configs/submission.yaml method/params differ from configs/dl/{stem}.yaml")
    if not weights.startswith("weights/") or Path(weights).is_absolute():
        raise SystemExit(f"unexpected weights path {weights!r}")
    header = CONFIG_HEADER.format(model=stem, level_mode=cfg["params"].get("level_mode", "none"))
    return (header + body).encode("utf-8"), weights


def clear_target(target: Path) -> None:
    def onerror(func, p, _exc):  # read-only files on Windows
        Path(p).chmod(stat.S_IWRITE)
        func(p)
    for p in target.iterdir():
        if p.name == ".git":
            continue
        if p.is_dir():
            shutil.rmtree(p, onerror=onerror)
        else:
            p.unlink()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="source tag (must be on --branch), e.g. r2-ready")
    ap.add_argument("--target", required=True, help="submission repo working tree, outside this repo")
    ap.add_argument("--branch", default="r2-prep", help="the tag must be an ancestor of this branch")
    ap.add_argument("--sub-tag", default="[태그]", help="README 제출 버전 tag (placeholder by default)")
    ap.add_argument("--sub-commit", default="[커밋]", help="README 제출 버전 commit (placeholder by default)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = ap.parse_args(argv)

    commit = git("rev-parse", f"{args.tag}^{{commit}}").decode().strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, args.branch], cwd=REPO_ROOT).returncode:
        raise SystemExit(f"{args.tag} ({commit[:7]}) is not on {args.branch}")
    target = Path(args.target).resolve()
    if target == REPO_ROOT or target.is_relative_to(REPO_ROOT) or REPO_ROOT.is_relative_to(target):
        raise SystemExit(f"target must be outside this repo: {target}")
    if not (target / ".git").exists():
        raise SystemExit(f"target is not a git clone (no .git): {target}")

    files: dict[str, bytes] = {p: blob(commit, p) for p in SOURCE_FILES}
    init = "src/nrc/methods/__init__.py"
    files[init] = patch_registry(files[init].decode("utf-8")).encode("utf-8")
    files["configs/submission.yaml"], weights = submission_config(commit)
    files[weights] = blob(commit, weights)
    files[".gitignore"] = GITIGNORE.encode()
    files["input/.gitkeep"] = b""
    readme = TEMPLATE.read_text(encoding="utf-8")
    files["README.md"] = (readme.replace("{{SUB_TAG}}", args.sub_tag)
                          .replace("{{SUB_COMMIT}}", args.sub_commit)).encode("utf-8")
    if "{{" in files["README.md"].decode("utf-8"):
        raise SystemExit("README template has unfilled placeholders")

    clear_target(target)
    for rel, data in sorted(files.items()):
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)

    head = git("rev-parse", "HEAD").decode().strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    rows = [f"{hashlib.sha256(d).hexdigest()}  {len(d):>9}  {rel}" for rel, d in sorted(files.items())]
    total = sum(len(d) for d in files.values())
    manifest = "\n".join([
        "# Submission repo MANIFEST (scripts/export_submission.py; not exported)",
        f"source_tag: {args.tag}",
        f"source_commit: {commit}",
        f"source_branch: {args.branch}",
        f"exporter_commit: {head}{' (dirty)' if dirty else ''}",
        f"exported_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"files: {len(files)}  total_bytes: {total}",
        "# sha256  bytes  path",
        *rows, ""])
    Path(args.manifest).write_text(manifest, encoding="utf-8", newline="\n")
    print(manifest)
    print(f"exported {len(files)} files ({total / 2 ** 20:.2f} MiB) from {args.tag} {commit[:7]} -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
