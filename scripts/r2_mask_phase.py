"""Mean |angle(M)| of a complex ratio mask M = M_r + j M_i on the validation noisy set (does the phase path act?).

For each crm model: angle(M) = atan2(M_i, M_r) per TF bin, reported as
  - plain mean |angle| over all bins (degrees),
  - |X|^2-weighted mean |angle| (bins that carry energy, degrees),
  - fraction of bins with |angle| > 5 deg, and mean |M_i| / mean M_r.
Magnitude-mask models have angle 0 by construction. Inference only, float input as read from disk.
Usage: python scripts/r2_mask_phase.py weights/gru_v6a.pt weights/gru_v7a.pt ... > logs/r2_v7_mask_phase.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nrc.models.gru_mask import load_weights  # noqa: E402


def main() -> int:
    torch.set_num_threads(4)
    files = sorted((ROOT / "data/noise_reduction_challenge/validation/noisy").glob("*.wav"))
    print(f"validation noisy files: {len(files)}")
    for w in sys.argv[1:]:
        m = load_weights(ROOT / w)
        if m.mask_type != "crm":
            print(f"{w}: mask_type {m.mask_type} -> angle(M) = 0")
            continue
        s_abs = s_w = s_wa = s_big = s_mi = s_mr = 0.0
        n = 0
        with torch.no_grad():
            for f in files:
                x, sr = sf.read(f, dtype="float32")
                spec = m.stft(torch.from_numpy(x)[None])
                m_r, m_i = m.masks(spec.abs())
                ang = torch.atan2(m_i, m_r).abs().double()
                e = (spec.abs() ** 2).double()
                s_abs += ang.sum().item(); n += ang.numel()
                s_wa += (ang * e).sum().item(); s_w += e.sum().item()
                s_big += (ang > np.deg2rad(5)).sum().item()
                s_mi += m_i.abs().double().sum().item(); s_mr += m_r.double().sum().item()
        print(f"{w}: imag_scale {m.mask_imag_scale}, mean |angle(M)| {np.rad2deg(s_abs / n):.3f} deg, "
              f"|X|^2-weighted {np.rad2deg(s_wa / s_w):.3f} deg, bins > 5 deg {s_big / n:.4f}, "
              f"mean |M_i| {s_mi / n:.4f}, mean M_r {s_mr / n:.4f} (bins {n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
