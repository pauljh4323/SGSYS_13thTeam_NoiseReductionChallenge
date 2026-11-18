"""Export the model weights of a training-state checkpoint (last.pt from scripts/train.py) to the
inference-only format of nrc.models.gru_mask.save_weights (model_cfg + model_state + epoch/pesq).
No training: the state dict is copied unchanged and checked to reload with weights_only=True.
Usage: python scripts/export_last.py <last.pt> <out.pt> [--pesq <val PESQ at that epoch, from train_log.csv>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nrc.models.gru_mask import GRUMask, load_weights, save_weights  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("last")
    ap.add_argument("out")
    ap.add_argument("--pesq", type=float, default=None)
    a = ap.parse_args()
    ck = torch.load(a.last, map_location="cpu", weights_only=True)
    model = GRUMask(**ck["cfg"]["model"])
    model.load_state_dict(ck["model"])  # strict
    save_weights(a.out, model, epoch=int(ck["epoch"]), pesq=a.pesq)
    re = load_weights(a.out)
    same = all(torch.equal(re.state_dict()[k], v) for k, v in ck["model"].items())
    print(f"{a.last} (epoch {ck['epoch']}) -> {a.out}; reload weights_only=True OK, state identical: {same}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
