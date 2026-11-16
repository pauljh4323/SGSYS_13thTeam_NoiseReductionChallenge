"""Insert an existing eval.py train per-file CSV as a sweep stage-reference point (same key as sweep.py).

Usage: sweep_inject_ref.py <repo> <sweep name> <config> <eval csv>. Only valid when the CSV was produced by
code bit-identical for that config (record which file in the experiment doc)."""
import json, sys
from pathlib import Path
import pandas as pd
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo / "scripts")); sys.path.insert(0, str(repo / "src"))
from _common import load_config
name, cfg_path, csv = sys.argv[2], sys.argv[3], sys.argv[4]
c = load_config(repo / cfg_path)
g = pd.read_csv(repo / csv)
key = json.dumps({"method": c["method"], "params": c["params"], "split": "train", "n": len(g)}, sort_keys=True, default=str)
by = g.groupby("target_snr_db").pesq.mean()
row = {"kind": "stage_reference", "name": c["name"], "method": c["method"], **{f"p_{k}": v for k, v in c["params"].items()},
       "pesq": g.pesq.mean(), **{f"pesq_snr{int(s)}": by.get(s) for s in (0.0, 5.0, 10.0, 15.0)},
       "si_sdr": g.si_sdr.mean(), "stoi": g.stoi.mean(), "failed": int(g.pesq.isna().sum()),
       "clipped": int(g.n_clipped.sum()), "n": len(g), "key": key}
logs = repo / "logs"
done = pd.read_csv(logs / f"sweep_{name}.csv")
done = pd.concat([done[done.key != key], pd.DataFrame([row])], ignore_index=True)
done.to_csv(logs / f"sweep_{name}.csv", index=False)
pd.DataFrame({"key": key, "segment_id": g.segment_id, "pesq": g.pesq}).to_csv(
    logs / f"sweep_{name}_perfile.csv", mode="a", header=False, index=False)
print("injected", c["name"], row["pesq"], len(g), "from", csv)
