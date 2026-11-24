"""Net2Net-style widening of a GRUMask checkpoint (Chen et al. 2016; exp/r2-dmix, gru_v8p2z4w init).

  python scripts/widen_gru.py --src weights/gru_v8p2z3d.pt --hidden 384 --out weights/gru_v8p2z3d_w384.pt

Builds the widened state (nrc.models.net2net.widen_state: old units copied, new units random incoming std 1e-2,
mean biases, all weights FROM new units 0), saves it with save_weights (loads with weights_only=True) and checks:
1. equivalence: widened vs source output (inference level_mode rms_norm path, nrc.methods.gru_mask.enhance) on
   --n-val stratified validation files (seed 0); max |diff| must be <= --tol (1e-5);
2. structure: every weight FROM a new unit is exactly 0;
3. gradient: on 4 rms-normalised train-style (clean, noisy) validation pairs, MSE(enhanced, clean) backward:
   grad on the outgoing (zero) weights of new units is non-zero at step 1; one Adam step (lr 5e-5), then the grad
   on the new units' incoming weights is non-zero (it is exactly 0 at step 1 because their outputs are muted).
Exit 1 if any check fails.
"""
from __future__ import annotations

import argparse
import copy
import sys

import numpy as np

from _common import REPO_ROOT

from nrc.io import load_manifest, read_row, resolve_data_root, stratified_subset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="weights/gru_v8p2z3d.pt")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--out", default="weights/gru_v8p2z3d_w384.pt")
    ap.add_argument("--std", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--target-rms", type=float, default=0.079130)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)

    import torch
    from nrc.methods.gru_mask import enhance
    from nrc.models.gru_mask import GRUMask, count_params, load_weights, save_weights
    from nrc.models.net2net import new_unit_masks, widen_state

    torch.set_num_threads(4)
    src_path, out_path = REPO_ROOT / args.src, REPO_ROOT / args.out
    ck = torch.load(src_path, map_location="cpu", weights_only=True)
    cfg = dict(ck["model_cfg"])
    h, layers = int(cfg["hidden"]), int(cfg["layers"])
    old = load_weights(src_path)
    cfg["hidden"] = args.hidden
    wide = GRUMask(**cfg)
    wide.load_state_dict(widen_state(ck["model_state"], h, args.hidden, args.std, args.seed))  # strict
    wide.eval()
    save_weights(out_path, wide, epoch=0, widened_from=str(args.src), widen_hidden=f"{h}->{args.hidden}",
                 widen_std=float(args.std), widen_seed=int(args.seed), config="gru_v8p2z3d_w384")
    ok = True
    reload = load_weights(out_path)  # weights_only=True
    print(f"src {args.src}: hidden {h}, {count_params(old):,} params; widened hidden {args.hidden}: "
          f"{count_params(wide):,} params; saved {args.out} ({out_path.stat().st_size / 2**20:.2f} MiB), "
          f"reloads with weights_only=True: {reload.config}")

    # 2. structure
    masks = new_unit_masks(h, args.hidden, layers)
    sd = wide.state_dict()
    nz = sum(int((sd[k][m.expand_as(sd[k])] != 0).sum()) for k, m in masks["out"].items())
    n_out = sum(int(m.expand_as(sd[k]).sum()) for k, m in masks["out"].items())
    print(f"structure: {n_out:,} weights from new units, non-zero {nz}")
    ok &= nz == 0

    # 1. equivalence on the inference path (rms_norm)
    root = resolve_data_root(args.data_root)
    df = stratified_subset(load_manifest(root, "validation"), args.n_val)
    kw = dict(level_norm=True, level_mode="rms_norm", target_rms=args.target_rms)
    diffs = []
    for _, r in df.iterrows():
        x = read_row(root, r, "noisy")
        diffs.append(float(np.max(np.abs(enhance(x, weights=args.out, **kw) - enhance(x, weights=args.src, **kw)))))
    mx = max(diffs)
    print(f"equivalence: {len(diffs)} validation files, max |widened - source| = {mx:.3e} "
          f"(median {np.median(diffs):.3e}), tol {args.tol:g} -> {'PASS' if mx <= args.tol else 'FAIL'}")
    ok &= mx <= args.tol

    # 3. gradient into the new units
    rows = df.iloc[:4]
    xs, cs = [], []
    for _, r in rows.iterrows():
        x, c = read_row(root, r, "noisy").astype(np.float64), read_row(root, r, "clean").astype(np.float64)
        g = args.target_rms / max(float(np.sqrt(np.mean(x ** 2))), 1e-6)
        xs.append(x * g)
        cs.append(c * g)
    n = min(len(v) for v in xs)
    xt = torch.tensor(np.stack([v[:n] for v in xs]), dtype=torch.float32)
    ct = torch.tensor(np.stack([v[:n] for v in cs]), dtype=torch.float32)
    model = copy.deepcopy(wide).train()
    params = dict(model.named_parameters())
    opt = torch.optim.Adam(model.parameters(), lr=5e-5)

    def grad_norms():
        opt.zero_grad()
        torch.mean((model(xt) - ct) ** 2).backward()
        res = {}
        for side in ("out", "in"):
            s = 0.0
            for k, m in masks[side].items():
                gr = params[k].grad
                s += float((gr[m.expand_as(gr)] ** 2).sum())
            res[side] = s ** 0.5
        return res

    g1 = grad_norms()
    opt.step()
    g2 = grad_norms()
    print(f"gradient step 1: |grad| outgoing weights of new units {g1['out']:.3e}, incoming {g1['in']:.3e} "
          f"(0 expected: outputs muted); after 1 Adam step (lr 5e-5): outgoing {g2['out']:.3e}, incoming "
          f"{g2['in']:.3e}")
    ok &= g1["out"] > 0 and g2["in"] > 0
    print("widen_gru:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
