"""Function-preserving widening of a GRUMask BiGRU (Net2WiderNet-style, Chen, Goodfellow & Shlens 2016; exp/r2-dmix).

widen_state(old_state, H, H2) maps a GRUMask state dict with hidden H to hidden H2 > H (training-only, not used by
main.py). Units 0..H-1 of every H-sized layer keep their old weights; units H..H2-1 are new:
- incoming weights (inp rows, GRU input and recurrent rows of every gate r/z/n) ~ N(0, std^2);
- biases = mean of the matching old bias block (per gate for the GRU, b_ih and b_hh separately; inp bias mean);
- every weight FROM a new unit is 0: GRU recurrent columns into old rows, GRU layer-0 input columns from new inp
  units, next-layer GRU input columns and out (Linear 2H -> F) columns from new units of either direction.
Old units therefore never see new units, and the output equals the old model's (up to float summation order).
New-to-new recurrent weights are random (part of the new unit's incoming weights).
Unlike Net2WiderNet's unit replication + output split, the new units start as zero-output random units, so the
gradient reaches their outgoing (zero) weights at step 1 and their incoming weights from step 2 on.
Bidirectional layer outputs are [fwd 0..H-1, bwd 0..H-1]; widened: [fwd old, fwd new, bwd old, bwd new].
"""
from __future__ import annotations

import re

import torch

_GRU_KEY = re.compile(r"^gru\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)(_reverse)?$")


def _bi_cols(old_in: int, h: int, h2: int) -> torch.Tensor:
    """Column index map for an input of size 2H (bidirectional concat) into 2*H2."""
    idx = torch.arange(old_in)
    return torch.where(idx < h, idx, idx + (h2 - h))


def _gate_rows(h: int, h2: int) -> torch.Tensor:
    """Row map of the 3 stacked gates (r, z, n): old row g*H + i -> g*H2 + i."""
    idx = torch.arange(3 * h)
    return (idx // h) * h2 + idx % h


def widen_state(old: dict, h: int, h2: int, std: float = 1e-2, seed: int = 0) -> dict:
    if h2 <= h:
        raise ValueError(f"new hidden {h2} must exceed old hidden {h}")
    g = torch.Generator().manual_seed(int(seed))

    def rnd(*shape, dtype):
        return (torch.randn(*shape, generator=g, dtype=torch.float64) * std).to(dtype)

    new, rows3 = {}, _gate_rows(h, h2)
    for k in sorted(old):  # sorted: deterministic random draws
        v = old[k]
        m = _GRU_KEY.match(k)
        if k == "inp.weight":                          # (H, Din) -> (H2, Din)
            w = torch.zeros(h2, v.shape[1], dtype=v.dtype)
            w[:h], w[h:] = v, rnd(h2 - h, v.shape[1], dtype=v.dtype)
            new[k] = w
        elif k == "inp.bias":
            b = torch.empty(h2, dtype=v.dtype)
            b[:h], b[h:] = v, v.mean()
            new[k] = b
        elif k == "out.weight":                        # (F, 2H) -> (F, 2H2), new columns 0
            w = torch.zeros(v.shape[0], 2 * h2, dtype=v.dtype)
            w[:, _bi_cols(2 * h, h, h2)] = v
            new[k] = w
        elif m:
            kind, layer = m.group(1), int(m.group(2))
            if kind.startswith("bias"):                # (3H) -> (3H2), new = per-gate mean of the old block
                b = torch.empty(3 * h2, dtype=v.dtype)
                for gi in range(3):
                    b[gi * h2:gi * h2 + h] = v[gi * h:(gi + 1) * h]
                    b[gi * h2 + h:(gi + 1) * h2] = v[gi * h:(gi + 1) * h].mean()
                new[k] = b
                continue
            if kind == "weight_hh":                    # (3H, H) -> (3H2, H2)
                cols, in2 = torch.arange(h), h2
            elif layer == 0:                           # input = inp output, H -> H2
                cols, in2 = torch.arange(h), h2
            else:                                      # input = previous bidirectional layer, 2H -> 2H2
                cols, in2 = _bi_cols(2 * h, h, h2), 2 * h2
            w = torch.zeros(3 * h2, in2, dtype=v.dtype)
            w[rows3[:, None], cols[None, :]] = v       # old rows: old columns copied, new columns stay 0
            for gi in range(3):                        # new rows: random over all columns
                w[gi * h2 + h:(gi + 1) * h2] = rnd(h2 - h, in2, dtype=v.dtype)
            new[k] = w
        else:                                          # H-independent (norm, conv, feature stats)
            new[k] = v.clone()
    return new


def new_unit_masks(h: int, h2: int, layers: int) -> dict:
    """Boolean masks over the widened state: 'out' = weights FROM new units (0 after widening), 'in' = incoming
    weights / biases of new units."""
    out_m, in_m = {}, {}
    new_r = torch.zeros(3 * h2, dtype=torch.bool)
    for gi in range(3):
        new_r[gi * h2 + h:(gi + 1) * h2] = True
    new_h = torch.arange(h2) >= h
    new_bi = torch.cat([new_h, new_h])
    in_m["inp.weight"] = new_h[:, None].expand(h2, -1)
    in_m["inp.bias"] = new_h
    out_m["out.weight"] = new_bi[None, :]
    for l in range(layers):
        for s in ("", "_reverse"):
            src = new_h if l == 0 else new_bi
            out_m[f"gru.weight_ih_l{l}{s}"] = (~new_r)[:, None] & src[None, :]
            out_m[f"gru.weight_hh_l{l}{s}"] = (~new_r)[:, None] & new_h[None, :]
            in_m[f"gru.weight_ih_l{l}{s}"] = new_r[:, None].expand(-1, src.numel())
            in_m[f"gru.weight_hh_l{l}{s}"] = new_r[:, None].expand(-1, h2)
            in_m[f"gru.bias_ih_l{l}{s}"] = new_r
            in_m[f"gru.bias_hh_l{l}{s}"] = new_r
    return {"out": out_m, "in": in_m}
