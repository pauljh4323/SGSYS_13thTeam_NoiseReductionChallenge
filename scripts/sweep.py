"""Grid sweep over method params using eval.py's scoring (in-memory 16-bit, as main.py writes).

Grid YAML: method, base_params (dict), grid (dict of lists) or grid_groups (list of such dicts,
expanded separately and concatenated), reference (list of config paths), and optionally
stage_reference (one config path).
Writes logs/sweep_<name>.csv sorted by PESQ; prints the top 5 + reference rows. Each finished
point is appended to that CSV immediately; a restart skips points already present (same key).
Per-file PESQ of every point goes to logs/sweep_<name>_perfile.csv (key, segment_id, pesq).
With stage_reference: every point also gets the paired comparison against that config on the same
files (d_mean, d_se = std/sqrt(n), d_z = d_mean/d_se, keep = d_mean >= 0.002 and >= 2*SE); the
stage adopts the keep=True grid point with the largest d_mean (else keeps the reference) and the
runner-up's d_mean +- SE is reported. The verdict is also written to logs/sweep_<name>_paired.txt.
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
from paired import keeps, paired_stats


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
    stage_ref = None
    if spec.get("stage_reference"):
        stage_ref = {**load_config(REPO_ROOT / spec["stage_reference"]), "kind": "stage_reference"}
        cfgs.append(stage_ref)
    for c in cfgs:
        get_method(c["method"], **c["params"])  # fail fast on bad params
    import pesq  # noqa: F401

    root = resolve_data_root(args.data_root)
    manifest = stratified_subset(load_manifest(root, args.split), args.subset)
    rows_meta = [r.to_dict() for _, r in manifest.iterrows()]
    logs = REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    out_csv = logs / f"sweep_{name}.csv"
    # Resume: each finished point is appended to out_csv at once; on restart, points whose
    # (method, params, split, n) key is already in the CSV are skipped.
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    perfile_csv = logs / f"sweep_{name}_perfile.csv"
    perfile = (pd.read_csv(perfile_csv) if perfile_csv.exists()
               else pd.DataFrame(columns=["key", "segment_id", "pesq"]))
    # A point counts as done only if its summary row AND its per-file rows are present.
    done_keys = (set(done["key"]) & set(perfile["key"])) if "key" in done.columns else set()

    def point_key(c):
        return json.dumps({"method": c["method"], "params": c["params"], "split": args.split,
                           "n": len(rows_meta)}, sort_keys=True, default=str)

    todo, seen = [], set(done_keys)
    for c in cfgs:  # also drops grid points identical to the stage reference (same key)
        if point_key(c) not in seen:
            seen.add(point_key(c))
            todo.append(c)
    if len(todo) < len(cfgs):
        print(f"resume: {len(cfgs) - len(todo)} of {len(cfgs)} points already in "
              f"{out_csv.relative_to(REPO_ROOT)} or identical to another point, skipped")
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
            if "key" in done.columns:  # re-run of a point whose per-file rows were missing
                done = done[done["key"] != row["key"]]
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(out_csv, index=False)
            pf = pd.DataFrame({"key": row["key"], "segment_id": g.segment_id, "pesq": g.pesq})
            perfile = pd.concat([perfile, pf], ignore_index=True)
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
    if stage_ref is not None:
        text = stage_verdict(name, spec, cfgs, stage_ref, point_key, perfile, grid_keys, fmt)
        print(text)
        (logs / f"sweep_{name}_paired.txt").write_text(text + "\n", encoding="utf-8")
    return 0


def stage_verdict(name, spec, cfgs, stage_ref, point_key, perfile, grid_keys, fmt) -> str:
    """Paired comparison of every grid point against the stage reference + keep-rule verdict."""
    perfile = perfile.drop_duplicates(["key", "segment_id"], keep="last")
    by_key = {k: g.set_index("segment_id").pesq for k, g in perfile.groupby("key")}
    ref_key = point_key(stage_ref)
    ref = by_key[ref_key]
    rows = []
    for c in cfgs:
        if c["kind"] != "grid":
            continue
        st = paired_stats(ref, by_key[point_key(c)])
        rows.append({"name": c["name"], **{f"p_{k}": c["params"].get(k) for k in grid_keys},
                     "pesq": by_key[point_key(c)].mean(), **st, "keep": keeps(st),
                     "is_ref": point_key(c) == ref_key, "params": c["params"]})
    t = pd.DataFrame(rows).sort_values("d_mean", ascending=False, ignore_index=True)
    cols = ["name"] + [f"p_{k}" for k in grid_keys] + ["pesq", "d_mean", "d_se", "d_z", "keep"]
    lines = [f"[paired vs stage reference {spec['stage_reference']} (pesq {ref.mean():.4f}, "
             f"n={ref.notna().sum()}); keep: d >= 0.002 and d >= 2*SE]",
             t[cols].to_string(**fmt)]
    cand = t[t.keep & ~t.is_ref]
    others = t[~t.is_ref]
    if len(cand):
        win = cand.iloc[0]
        runner = others[others.name != win["name"]].head(1)
        changed = {k: win["params"][k] for k in grid_keys
                   if win["params"].get(k) != stage_ref["params"].get(k)}
        edges = []
        for k in grid_keys:
            vals = [v for g in (spec.get("grid_groups") or [spec.get("grid", {})]) for v in g.get(k, [])]
            if len(vals) > 2 and all(isinstance(v, (int, float)) for v in vals):
                if win["params"][k] in (min(vals), max(vals)):
                    edges.append(f"{k}={win['params'][k]} at grid edge")
        lines.append(f"VERDICT {name}: ADOPT {win['name']} {changed} "
                     f"d={win.d_mean:+.4f} +- {win.d_se:.4f} (z {win.d_z:.2f})")
        if len(runner):
            r = runner.iloc[0]
            lines.append(f"  runner-up {r['name']} {({k: r['params'][k] for k in grid_keys})} "
                         f"d={r.d_mean:+.4f} +- {r.d_se:.4f} (z {r.d_z:.2f}, keep={r.keep})")
        lines.append("  grid edge: " + ("; ".join(edges) if edges else "none"))
    else:
        best = others.head(1)
        lines.append(f"VERDICT {name}: KEPT reference"
                     + (f" (best {best.iloc[0]['name']} d={best.iloc[0].d_mean:+.4f} +- "
                        f"{best.iloc[0].d_se:.4f}, z {best.iloc[0].d_z:.2f})" if len(best) else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
