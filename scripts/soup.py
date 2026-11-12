"""Uniform model soup (Wortsman et al. 2022): average the weights of fine-tuned gru_mask checkpoints.

python scripts/soup.py --weights weights/A.pt weights/B.pt ... --out weights/<name>.pt

All checkpoints must share format, model_cfg and state_dict keys/shapes/dtypes. Float parameters and
float buffers are averaged uniformly (in float64, cast back); non-float buffers come from the first
model. BatchNorm running stats would be stale after averaging; recalibration is only needed for
BatchNorm models and is not implemented, so a BatchNorm architecture is rejected. The gru_mask conv2d
front end uses GroupNorm by default (no running stats), so nothing needs recalibration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

from _common import REPO_ROOT  # noqa: F401  # sets sys.path for nrc

from nrc.models.gru_mask import GRUMask, save_weights


def soup_state(ckpts: list[dict]) -> dict:
    first = ckpts[0]
    for i, c in enumerate(ckpts[1:], 1):
        assert c.get("format") == first.get("format"), f"format mismatch in checkpoint {i}"
        assert c["model_cfg"] == first["model_cfg"], f"model_cfg mismatch in checkpoint {i}"
        assert list(c["model_state"]) == list(first["model_state"]), f"state_dict keys differ in {i}"
    out = {}
    for k, v0 in first["model_state"].items():
        vs = [c["model_state"][k] for c in ckpts]
        for i, v in enumerate(vs):
            assert v.shape == v0.shape and v.dtype == v0.dtype, f"{k}: shape/dtype mismatch in {i}"
        if v0.is_floating_point():
            out[k] = (sum(v.double() for v in vs) / len(vs)).to(v0.dtype)
        else:
            out[k] = v0.clone()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ckpts = [torch.load(p, map_location="cpu", weights_only=True) for p in args.weights]
    state = soup_state(ckpts)
    model = GRUMask(**ckpts[0]["model_cfg"])
    model.load_state_dict(state)
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise SystemExit("BatchNorm present: running-stat recalibration required but not implemented")
    print("no BatchNorm (GroupNorm front end): recalibration skipped")
    for k in ("feat_mean", "feat_std"):
        spread = max(float((c["model_state"][k] - ckpts[0]["model_state"][k]).abs().max()) for c in ckpts)
        print(f"{k}: max |diff vs first| across sources = {spread:.3g}")
    out = Path(args.out)
    sources = [Path(p).name for p in args.weights]
    save_weights(out, model.eval(), config=out.stem, soup="uniform", soup_sources=sources,
                 bn_recalibrated=False)
    print(f"wrote {out} ({len(sources)} sources: {', '.join(sources)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
