"""Grid sweep over method params using eval.py's scoring (in-memory 16-bit, as main.py writes).

Grid YAML: method, base_params (dict), grid (dict of lists) or grid_groups (list of such dicts,
expanded separately and concatenated), reference (list of config paths).
Writes logs/sweep_<name>.csv sorted by PESQ; prints the top 5 + reference rows. Each finished
point is appended to that CSV immediately; a restart skips points already present (same key).
--per-file also appends each point's per-file rows (name, segment_id, target_snr_db,
noise_difficulty, pesq, snr_db) to logs/sweep_<name>_perfile.csv.
Does NOT append to results/experiments.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from _common import REPO_ROOT, load_config

from eval import evaluate_row
from nrc.io import load_manifest, resolve_data_root, stratified_subset
from nrc.methods import get_method


def grid_points(spec: dict) -> list[dict]:
    """Cartesian product of `grid`, or the concatenation of each product in `grid_groups`."""
    base = dict(spec.get("base_params", {}))
    groups = spec.get("grid_groups") or [spec.get("grid", {})]
    return [{**base, **dict(zip(g, vals))}
            for g in groups for vals in itertools.product(*(g[k] for k in g))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", required=True, help="configs/sweeps/<name>.yaml")
    ap.add_argument("--split", default="validation", choices=("train", "validation"))
    ap.add_argument("--subset", type=int, default=32)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--per-file", action="store_true",
                    help="also append per-file rows to logs/sweep_<name>_perfile.csv")
    args = ap.parse_args(argv)

    grid_path = Path(args.grid)
    with open(grid_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    name = grid_path.stem
    cfgs = [{"name": f"{name}_{i:03d}", "method": spec["method"], "params": p, "kind": "grid"}
            for i, p in enumerate(grid_points(spec))]
    for ref in spec.get("reference", []):
        c = load_config(REPO_ROOT / ref)
        cfgs.append({**c, "kind": "reference"})
    for c in cfgs:
        get_method(c["method"], **c["params"])  # fail fast on bad params
    import pesq  # noqa: F401

    root = resolve_data_root(args.data_root)
    manifest = stratified_subset(load_manifest(root, args.split), args.subset)
    rows_meta = [r.to_dict() for _, r in manifest.iterrows()]
    logs = REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    out_csv = logs / f"sweep_{name}.csv"
    perfile_csv = logs / f"sweep_{name}_perfile.csv"
    # Resume: each finished point is appended to out_csv at once; on restart, points whose
    # (method, params, split, n) key is already in the CSV are skipped.
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    done_keys = set(done["key"]) if "key" in done.columns else set()

    def point_key(c):
        return json.dumps({"method": c["method"], "params": c["params"], "split": args.split,
                           "n": len(rows_meta)}, sort_keys=True, default=str)

    todo = [c for c in cfgs if point_key(c) not in done_keys]
    if len(todo) < len(cfgs):
        print(f"resume: {len(cfgs) - len(todo)} of {len(cfgs)} points already in "
              f"{out_csv.relative_to(REPO_ROOT)}, skipped")
    t0 = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
    try:
        for c in todo:
            tasks = [(root, r, {k: c[k] for k in ("name", "method", "params")},
                      args.split, "mem16", None) for r in rows_meta]
            results = (list(pool.map(evaluate_row, tasks, chunksize=4)) if pool
                       else [evaluate_row(t) for t in tasks])
            g = pd.DataFrame(results)
            by_snr = g.groupby("target_snr_db").pesq.mean()
            row = {"kind": c["kind"], "name": c["name"], "method": c["method"],
                   **{f"p_{k}": v for k, v in c["params"].items()},
                   "pesq": g.pesq.mean(),
                   **{f"pesq_snr{int(s)}": by_snr.get(s) for s in (0.0, 5.0, 10.0, 15.0)},
                   "si_sdr": g.si_sdr.mean(), "stoi": g.stoi.mean(),
                   "failed": int(g.pesq.isna().sum()), "clipped": int(g.n_clipped.sum()),
                   "n": len(g), "key": point_key(c)}
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(out_csv, index=False)
            if args.per_file:
                pf = g[["segment_id", "target_snr_db", "noise_difficulty", "pesq", "snr_db"]]
                pf.insert(0, "name", c["name"])
                pf.to_csv(perfile_csv, mode="a", header=not perfile_csv.exists(), index=False)
            print(f"  done {c['name']}: pesq {row['pesq']:.4f}", flush=True)
    finally:
        if pool:
            pool.shutdown()
    elapsed = time.perf_counter() - t0

    keys = {point_key(c) for c in cfgs}
    table = (done[done["key"].isin(keys)] if "key" in done.columns else done)
    table = table.sort_values("pesq", ascending=False, ignore_index=True)
    done.sort_values("pesq", ascending=False, ignore_index=True).to_csv(out_csv, index=False)

    grid_keys = dict.fromkeys(k for g in (spec.get("grid_groups") or [spec.get("grid", {})])
                              for k in g)
    show = (["name"] + [f"p_{k}" for k in grid_keys]
            + ["pesq", "pesq_snr0", "pesq_snr5", "pesq_snr10", "pesq_snr15", "si_sdr", "failed"])
    fmt = {"float_format": lambda v: f"{v:.4f}", "index": False}
    print(f"sweep {name}: {len(cfgs)} configs x {len(rows_meta)} files | {args.split} | "
          f"16-bit | {elapsed:.1f}s -> {out_csv.relative_to(REPO_ROOT)}")
    print("[top 5]\n" + table[table.kind == "grid"].head(5)[show].to_string(**fmt))
    print("[reference]\n" + table[table.kind == "reference"][show].to_string(**fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
