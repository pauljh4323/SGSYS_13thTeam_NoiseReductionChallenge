"""Shared helpers for scripts/: sys.path setup, config loading, common CLI args."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402


def load_config(path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "method" not in cfg:
        raise ValueError(f"{path}: missing 'method'")
    cfg.setdefault("params", {})
    cfg["name"] = path.stem
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="configs/<name>.yaml")
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--subset", type=int, default=None,
                        help="fixed stratified subset of N files (seed 0)")
    parser.add_argument("--data-root", default=None)


def output_dir(cfg_name: str, split: str) -> Path:
    return REPO_ROOT / "outputs" / cfg_name / split


def output_path(cfg_name: str, split: str, noisy_rel: str) -> Path:
    """outputs/<config>/<split>/<noisy path relative to the split folder>."""
    rel = Path(noisy_rel)
    if rel.parts and rel.parts[0] == split:
        rel = Path(*rel.parts[1:])
    return output_dir(cfg_name, split) / rel


def git_hash() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        # results/experiments.csv is appended by eval.py itself; it must not mark later rows dirty.
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--",
                                ".", ":!results/experiments.csv"],
                               cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
