"""dsp-p unattended pipeline: P2a-bc wait -> P2a edge extensions -> P2b -> P3 (+ extensions) -> finalist evals.

Launched detached (survives the interactive session). Re-running it resumes: sweeps skip finished points
(sweep.py keys) and points already evaluated in any logs/sweep_dsp_p_*.csv are copied, not re-run.
Every step appends one line to logs/pipeline_dsp_p.status; the last line is DONE (or DEAD / FAILED).
Selection: paired keep rule on the FULL train set (d >= 0.002 and >= 2 SE, scripts/paired.py), --jobs 2.
Before the finalist evals configs/dsp_p.yaml and the generated sweep grids are committed (local commit on the
current branch, paths limited to those files) so eval.py rows get a clean hash; no merge, tag, push or e2e.
"""
from __future__ import annotations

import glob
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from _common import REPO_ROOT, git_hash, load_config
from paired import keeps, paired_stats

JOBS = 2
N_TRAIN = 1296
LOGS = REPO_ROOT / "logs"
STATUS = LOGS / "pipeline_dsp_p.status"
SWEEPS = REPO_ROOT / "configs" / "sweeps"
PY = sys.executable
BASE_CFG = "configs/dsp_k.yaml"
MAX_EXT = 4  # extension steps per parameter


def status(step, adopted="", stats=None, note=""):
    d = (f"d={stats['d_mean']:+.4f} +- {stats['d_se']:.4f} (z {stats['d_z']:.2f})" if stats else "")
    line = " | ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, str(adopted), d, note])
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def key_of(params):
    return json.dumps({"method": "mmse_lsa", "params": params, "split": "train", "n": N_TRAIN},
                      sort_keys=True, default=str)


def perfile_index():
    """key -> (summary row, per-file DataFrame) over every dsp_p sweep log (incl. the dsp_k reference)."""
    idx = {}
    for csv in sorted(glob.glob(str(LOGS / "sweep_dsp_p_*.csv"))):
        if csv.endswith("_perfile.csv"):
            continue
        s = pd.read_csv(csv)
        pf = pd.read_csv(csv[:-4] + "_perfile.csv")
        for _, r in s.iterrows():
            rows = pf[pf.key == r.key].drop_duplicates("segment_id", keep="last")
            if len(rows) == N_TRAIN and r.key not in idx:
                idx[r.key] = (r, rows)
    return idx


def series(params, idx=None):
    idx = idx or perfile_index()
    return idx[key_of(params)][1].set_index("segment_id").pesq


def copy_known(name, cfgs):
    """Copy already-evaluated points (same key) into logs/sweep_<name>*.csv so sweep.py skips them."""
    idx = perfile_index()
    out, outpf = LOGS / f"sweep_{name}.csv", LOGS / f"sweep_{name}_perfile.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["key"])
    add, addpf = [], []
    for cname, kind, params in cfgs:
        k = key_of(params)
        if k in idx and k not in set(done.key):
            row = idx[k][0].copy()
            row["name"], row["kind"] = cname, kind
            add.append(row)
            addpf.append(idx[k][1][["key", "segment_id", "pesq"]])
    if add:
        pd.concat([done, pd.DataFrame(add)], ignore_index=True).to_csv(out, index=False)
        pd.concat(addpf).to_csv(outpf, mode="a", index=False, header=not outpf.exists())


def run_sweep(name, base, grid, ref_params, ref_name, comment):
    """Write the grid + stage-reference config, run sweep.py, return (adopted params, stats, table)."""
    ref_path = LOGS / f"{ref_name}.yaml"
    ref_path.write_text(yaml.safe_dump({"method": "mmse_lsa", "params": ref_params}, sort_keys=False),
                        encoding="utf-8")
    spec = {"method": "mmse_lsa", "base_params": base, "grid": grid,
            "stage_reference": ref_path.relative_to(REPO_ROOT).as_posix()}
    (SWEEPS / f"{name}.yaml").write_text(
        "".join(f"# {c}\n" for c in comment) + yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    points = [{**base, **dict(zip(grid, v))} for v in itertools.product(*grid.values())]
    copy_known(name, [(ref_name, "stage_reference", ref_params)]
               + [(f"{name}_{i:03d}", "grid", p) for i, p in enumerate(points)])
    with open(LOGS / f"sweep_{name}_run.txt", "a", encoding="utf-8") as log:
        rc = subprocess.run([PY, "scripts/sweep.py", "--grid", f"configs/sweeps/{name}.yaml",
                             "--split", "train", "--subset", str(N_TRAIN), "--jobs", str(JOBS)],
                            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        status(f"{name} FAILED", note=f"sweep.py exit {rc}, see logs/sweep_{name}_run.txt")
        status("FAILED")
        sys.exit(1)
    return verdict(name, points, ref_params)


def verdict(name, points, ref_params):
    idx = perfile_index()
    ref = series(ref_params, idx)
    table = []
    for i, p in enumerate(points):
        if key_of(p) == key_of(ref_params):
            continue
        st = paired_stats(ref, series(p, idx))
        table.append((f"{name}_{i:03d}", p, st, keeps(st)))
    table.sort(key=lambda t: -t[2]["d_mean"])
    win = next((t for t in table if t[3]), None)
    return (win[1], win[2], table) if win else (None, table[0][2] if table else None, table)


def changed(p, ref):
    return {k: v for k, v in p.items() if ref.get(k) != v}


def report_stage(step, ref, adopted, st, table, name):
    runner = next((t for t in table if adopted is None or t[1] != adopted), None)
    rnote = (f"runner-up {changed(runner[1], ref)} d={runner[2]['d_mean']:+.4f} +- {runner[2]['d_se']:.4f}"
             if runner else "")
    if adopted is None:
        status(step, "KEPT reference", st, "best " + rnote.replace("runner-up ", ""))
    else:
        status(step, changed(adopted, ref), st, rnote)
        bd = LOGS / f"breakdown_{name}.txt"
        win_name = next(t[0] for t in table if t[1] == adopted)
        subprocess.run([PY, "scripts/breakdown.py", "--sweep", name, "--a", ref_name_of(name), "--b", win_name,
                        "--out", bd.relative_to(REPO_ROOT).as_posix()], cwd=REPO_ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ref_name_of(name):
    return "dsp_k" if name == "dsp_p_p2a_bc" else f"dsp_p_ref_{name.removeprefix('dsp_p_')}"


def extend(stage, cur, rules, tested):
    """One-step-past-the-edge extensions, one parameter at a time, until interior or losing."""
    n = 0
    progress = True
    counts = {k: 0 for k in rules}
    while progress:
        progress = False
        for k, nxt in rules.items():
            if counts[k] >= MAX_EXT:
                continue
            nv = nxt(cur[k], tested[k])
            if nv is None or nv in tested[k]:
                continue
            n += 1
            counts[k] += 1
            name = f"dsp_p_{stage}_x{n}"
            tested[k].add(nv)
            adopted, st, table = run_sweep(
                name, cur, {k: [nv]}, cur, ref_name_of(name),
                [f"dsp-p {stage} edge extension {n}: {k} {cur[k]} -> {nv} (pipeline, paired vs current adopted)."])
            report_stage(f"{stage} ext {k}={nv}", cur, adopted, st if adopted else table[0][2], table, name)
            if adopted is not None:
                cur = adopted
                progress = True
    return cur


def wait_p2a_bc():
    done = LOGS / "sweep_dsp_p_p2a_bc.done"
    while not done.exists():
        ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object "
                             "{ $_.CommandLine -like '*sweep.py*dsp_p_p2a_bc*' }).Count"],
                            capture_output=True, text=True).stdout.strip()
        if ps in ("", "0"):
            time.sleep(10)
            if done.exists():
                break
            n = len(pd.read_csv(LOGS / "sweep_dsp_p_p2a_bc.csv").query("kind == 'grid'"))
            status("P2a-bc wait DEAD", note=f"no sweep.py process and no done marker; {n}/24 points in csv")
            status("DEAD")
            sys.exit(1)
        time.sleep(60)


def finalist(cur, history):
    base = load_config(REPO_ROOT / BASE_CFG)["params"]
    head = ["dsp-p finalist (scripts/run_pipeline_dsp_p.py; staged paired sweeps on the FULL train set, 1296 files,",
            "keep rule d >= 0.002 and >= 2*SE). = dsp-v7 (dsp_k) plus:"] + [f"  {h}" for h in history] + [
            "Logs: logs/pipeline_dsp_p.status, logs/sweep_dsp_p_*_paired.txt, docs/exp_dsp_p.md."]
    (REPO_ROOT / "configs" / "dsp_p.yaml").write_text(
        "".join(f"# {h}\n" for h in head)
        + yaml.safe_dump({"method": "mmse_lsa", "params": {**base, **cur}}, sort_keys=False), encoding="utf-8")
    paths = ["configs/dsp_p.yaml"] + sorted(
        p.relative_to(REPO_ROOT).as_posix() for p in SWEEPS.glob("dsp_p_*.yaml"))
    subprocess.run(["git", "add", "--", *paths], cwd=REPO_ROOT)
    subprocess.run(["git", "commit", "-q", "-m",
                    "dsp-p: finalist configs/dsp_p.yaml and pipeline sweep grids (pipeline commit)\n\n"
                    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", *paths], cwd=REPO_ROOT)
    h = git_hash()
    status("finalist config committed", changed({**base, **cur}, base), note=f"hash {h}")
    out = []
    for split in ("train", "validation", "val_remix"):
        subprocess.run([PY, "scripts/eval.py", "--config", "configs/dsp_p.yaml", "--split", split,
                        "--jobs", str(JOBS)], cwd=REPO_ROOT,
                       stdout=open(LOGS / f"pipeline_dsp_p_eval_{split}.txt", "w", encoding="utf-8"),
                       stderr=subprocess.STDOUT)
        a = pd.read_csv(REPO_ROOT / "outputs" / "dsp_k" / f"eval_{split}.csv").set_index("segment_id")
        b = pd.read_csv(REPO_ROOT / "outputs" / "dsp_p" / f"eval_{split}.csv").set_index("segment_id")
        st = paired_stats(a.pesq, b.pesq)
        status(f"finalist {split}", f"PESQ {b.pesq.mean():.4f} (dsp_k {a.pesq.mean():.4f}) SNR {b.snr_db.mean():.2f} "
               f"(dsp_k {a.snr_db.mean():.2f})", st)
        out.append(f"[{split}] dsp_p PESQ {b.pesq.mean():.4f} SNR {b.snr_db.mean():.2f} | dsp_k PESQ "
                   f"{a.pesq.mean():.4f} SNR {a.snr_db.mean():.2f} | paired d={st['d_mean']:+.4f} +- "
                   f"{st['d_se']:.4f} (z {st['d_z']:.2f})")
        for by in ("target_snr_db", "noise_difficulty"):
            ga, gb = a.groupby(by), b.groupby(by)
            out.append(f"  by {by}: " + ", ".join(
                f"{g}: {gb.get_group(g).pesq.mean():.3f}/{ga.get_group(g).pesq.mean():.3f} "
                f"SNR {gb.get_group(g).snr_db.mean():.2f}" for g in sorted(b[by].unique())))
        subprocess.run([PY, "scripts/breakdown.py", "--csv", f"outputs/dsp_k/eval_{split}.csv",
                        f"outputs/dsp_p/eval_{split}.csv", "--split", split,
                        "--out", f"logs/breakdown_dsp_p_finalist_{split}.txt"], cwd=REPO_ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    probe = []
    for g in (0, -10, -20, -30):
        subprocess.run([PY, "scripts/eval.py", "--config", "configs/dsp_p.yaml", "--split", "validation",
                        "--jobs", str(JOBS), "--input-gain-db", str(g)], cwd=REPO_ROOT,
                       stdout=open(LOGS / f"pipeline_dsp_p_probe_{g}dB.txt", "w", encoding="utf-8"),
                       stderr=subprocess.STDOUT)
        pa = pd.read_csv(REPO_ROOT / "outputs" / "dsp_k" / f"eval_validation_gain{g}dB.csv").pesq.mean()
        pb = pd.read_csv(REPO_ROOT / "outputs" / "dsp_p" / f"eval_validation_gain{g}dB.csv").pesq.mean()
        probe.append(f"{g} dB: {pb:.4f} vs dsp_k {pa:.4f} (diff {pb - pa:+.4f}, gate >= -0.02: {pb - pa >= -0.02})")
    status("finalist probe", "; ".join(probe))
    (LOGS / "pipeline_dsp_p_finalist.txt").write_text("\n".join(out + ["[probe]"] + probe) + "\n", encoding="utf-8")


def main():
    status("pipeline start", note=f"pid {__import__('os').getpid()} hash {git_hash()}")
    wait_p2a_bc()
    base = load_config(REPO_ROOT / BASE_CFG)["params"]
    spec = yaml.safe_load(open(SWEEPS / "dsp_p_p2a_bc.yaml", encoding="utf-8"))
    points = [{**spec["base_params"], **dict(zip(spec["grid"], v))}
              for v in itertools.product(*spec["grid"].values())]
    adopted, st, table = verdict("dsp_p_p2a_bc", points, base)
    report_stage("P2a-bc (log_bc) vs dsp_k", base, adopted, st, table, "dsp_p_p2a_bc")
    history = []
    if adopted is None:
        cur = dict(base)
    else:
        tested = {"nlm_search_t": set(spec["grid"]["nlm_search_t"]), "nlm_h": set(spec["grid"]["nlm_h"])}
        cur = extend("p2a", dict(adopted), {
            "nlm_search_t": lambda v, t: 2 * v if v == max(t) else None,
            "nlm_h": lambda v, t: 2 * v if v == max(t) else None}, tested)
        history.append("P2a: " + json.dumps(changed(cur, base)) + f" (d vs dsp_k {paired_stats(series(base), series(cur))['d_mean']:+.4f})")
        status("P2a final", changed(cur, base), paired_stats(series(base), series(cur)))
        # P2b nlm_mix at the adopted P2a point (reference nlm_mix 0.5)
        name = "dsp_p_p2b"
        mix_vals = [0.3, 0.6, 1.0]
        adopted, st, table = run_sweep(name, cur, {"nlm_mix": mix_vals}, cur, ref_name_of(name),
                                       ["dsp-p P2b: nlm_mix at the adopted P2a point (pipeline, paired vs nlm_mix 0.5)."])
        report_stage("P2b nlm_mix", cur, adopted, st, table, name)
        if adopted is not None:
            cur = adopted
        tested = {"nlm_mix": set(mix_vals) | {0.5}}
        cur = extend("p2b", cur, {
            "nlm_mix": lambda v, t: 0.8 if v == 1.0 else (round(v / 2, 3) if v == min(t) else None)}, tested)
        history.append(f"P2b: nlm_mix {cur['nlm_mix']}")
    # P3 gain smoothing
    name = "dsp_p_p3"
    p3_base = {**cur, "gain_smooth": True}
    grid = {"gs_thr_db": [-5.0, 0.0, 5.0], "gs_bins": [1, 2], "gs_frames": [0, 1]}
    adopted, st, table = run_sweep(name, p3_base, grid, cur, ref_name_of(name),
                                   ["dsp-p P3: low-SNR 2D gain smoothing at the adopted P2 point (pipeline, paired vs P2)."])
    report_stage("P3 gain_smooth", cur, adopted, st, table, name)
    if adopted is not None:
        tested = {k: set(v) for k, v in grid.items()}
        cur = extend("p3", adopted, {
            "gs_thr_db": lambda v, t: v + 5.0 if v == max(t) else (v - 5.0 if v == min(t) else None),
            "gs_bins": lambda v, t: v + 1 if v == max(t) else None,
            "gs_frames": lambda v, t: v + 1 if v == max(t) else None}, tested)
        history.append("P3: " + json.dumps({k: cur[k] for k in ("gain_smooth", "gs_thr_db", "gs_bins", "gs_frames")}))
    else:
        history.append("P3: gain_smooth not adopted")
    st = paired_stats(series(base), series(cur))
    status("selection final vs dsp_k (train)", changed(cur, base), st)
    finalist(cur, history)
    status("DONE")


if __name__ == "__main__":
    main()
